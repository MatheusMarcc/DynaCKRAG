"""Controladores do laço de evidência.

* ``LearnedController`` — o controlador aprendido do DynaKRAG:
      a*_t = argmax_{a ∈ A(s_t)} [ v̂θ(s_t, a) − λ·c(s_t, a) ]              (Eq. 2, λ = 0 nas rodadas principais)
  e um **modelo de continuação** separado que "can suppress stop_answer when additional evidence is
  predicted to help": se o argmax for parar e P(continuar ajuda | s_t) ≥ τ, o controlador escolhe a
  melhor ação que não é parar. τ baixo = "permissive continuation thresholds".
* ``UniformController`` — a ablação "without learned controller": escolha uniforme (com semente) entre
  as ações válidas, horizonte fixo (só para no limite), sem continuação aprendida.

Quando o controlador pode parar. Os alvos da Eq. (1) dão a ``stop_answer`` o valor SR(s), enquanto as
ações de aquisição valem ΔSR (incrementos): assim que há alguma evidência, o argmax quase sempre é parar.
Por isso o artigo põe a parada sob uma lógica própria (Tabela 5: stop_answer é "enabled by the stopping and
continuation logic or by a hard cap"), em duas camadas:

1. a camada de validade (``validity.py``, ``stop_rule``) só deixa ``stop_answer`` entrar em A(s_t) depois que
   a sonda de suficiência disse "suficiente" — é o que as Tabelas 6–8 do artigo mostram: sem a sonda nenhum
   exemplo para antes do limite, e com ela só param cedo os que saem logo depois da primeira busca;
2. com a parada válida e no topo do ranking, o modelo de continuação ainda pode segurá-la (P ≥ τ).

Sem modelo de continuação treinado (rótulos de uma classe só), a camada 1 decide sozinha; com a regra antiga
``stop_rule: always`` a reserva ``sufficient_only`` faz o mesmo papel.
"""
from __future__ import annotations

import os
import pickle
import random

from .features import CONT_FEATURE_NAMES, VALUE_FEATURE_NAMES, continuation_features, value_features
from .state import STOP, EvidenceState, control_slug, scalar_cost


class LearnedController:
    name = "learned"

    def __init__(self, value_model, cont_model=None, lam: float = 0.0, tau: float = 0.35,
                 use_continuation: bool = True, max_steps: int = 4, retrieval_cap: int | None = None,
                 allow_stop: bool = True, fallback_gate: str = "sufficient_only"):
        self.value_model = value_model
        self.cont_model = cont_model
        self.lam = lam
        self.tau = tau
        self.use_continuation = use_continuation and cont_model is not None
        # sem modelo de continuação: regra de reserva (só se o uso de continuação foi pedido)
        self.fallback_gate = fallback_gate if (use_continuation and cont_model is None) else None
        self.max_steps = max_steps
        self.retrieval_cap = retrieval_cap
        self.allow_stop = allow_stop

    @property
    def gate(self) -> str:
        if self.use_continuation:
            return f"continuation(τ={self.tau:.2f})"
        return self.fallback_gate or "none"

    def scores(self, s: EvidenceState, A: list[str], env) -> dict[str, float]:
        X = value_features(s, A, env, self.max_steps, self.retrieval_cap)
        v = self.value_model.predict(X)
        return {a: float(v[i]) - self.lam * scalar_cost(a) for i, a in enumerate(A)}

    def p_continue(self, s: EvidenceState, env) -> float:
        x = continuation_features(s, env, self.max_steps, self.retrieval_cap).reshape(1, -1)
        proba = self.cont_model.predict_proba(x)[0]
        classes = list(self.cont_model.classes_)
        return float(proba[classes.index(1)]) if 1 in classes else 0.0

    def choose(self, s: EvidenceState, A: list[str], env) -> tuple[str, dict]:
        if A == [STOP]:
            return STOP, {"forced": True}
        cand = A if self.allow_stop else [a for a in A if a != STOP]
        sc = self.scores(s, cand, env)
        best = max(cand, key=lambda a: (sc[a], -cand.index(a)))
        info = {"scores": {a: round(v, 4) for a, v in sc.items()}}
        others = [a for a in cand if a != STOP]
        if best == STOP and others:
            suppress = False
            if self.use_continuation:
                p = self.p_continue(s, env)
                info["p_continue"] = round(p, 4)
                suppress = p >= self.tau
            elif self.fallback_gate == "sufficient_only":
                suppress = s.suff != 1
                info["gate"] = "sufficient_only"
            if suppress:
                best = max(others, key=lambda a: (sc[a], -others.index(a)))
                info["stop_suppressed"] = True
        return best, info


class UniformController:
    name = "uniform"

    def __init__(self, seed: int = 13):
        self.seed = seed

    def choose(self, s: EvidenceState, A: list[str], env) -> tuple[str, dict]:
        non_stop = [a for a in A if a != STOP]
        if not non_stop:
            return STOP, {"forced": True}
        rng = random.Random(f"{self.seed}-{s.qid}-{s.steps}")
        return rng.choice(non_stop), {}


# ----------------------------------------------------------------------------- persistência
def controller_path(cfg: dict, dataset: str) -> str:
    """Um arquivo por (nº de exemplos de treino, regras de controle): trocar as regras treina outro controlador
    e deixa o antigo no lugar."""
    name = f"controller_n{cfg['dynakrag']['n_train']}{control_slug(cfg)}.pkl"
    return os.path.join(cfg["paths"]["data"], dataset, "dynakrag", name)


def save_models(path: str, value_model, cont_model, meta: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"value": value_model, "cont": cont_model, "meta": meta,
                     "value_features": VALUE_FEATURE_NAMES, "cont_features": CONT_FEATURE_NAMES}, f)


def load_models(path: str) -> dict:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if obj.get("value_features") != VALUE_FEATURE_NAMES:
        raise RuntimeError(f"{path} foi treinado com outras features — rode `dynakrag train --force`")
    return obj


def make_controller(cfg: dict, dataset: str, kind: str = "learned", retrieval_cap: int | None = None,
                    use_continuation: bool | None = None):
    dk = cfg["dynakrag"]
    if kind == "uniform":
        return UniformController(int(cfg.get("seed", 13)))
    obj = load_models(controller_path(cfg, dataset))
    cc = dk["controller"]
    tau = cc.get("continuation_threshold", "auto")
    if tau in (None, "auto"):
        tau = (obj.get("meta", {}).get("continuation", {}) or {}).get("tau") or 0.35
    return LearnedController(obj["value"], obj.get("cont"), lam=float(dk.get("lambda_cost", 0.0)),
                             tau=float(tau),
                             use_continuation=cc.get("continuation", True) if use_continuation is None else use_continuation,
                             max_steps=int(dk["max_steps"]), retrieval_cap=retrieval_cap)


__all__ = ["LearnedController", "UniformController", "controller_path", "save_models", "load_models",
           "make_controller"]
