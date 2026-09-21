"""Políticas do laço A.

* UniformPolicy — escolha uniforme (com semente) entre as ações válidas, horizonte fixo.
  É exatamente a ablação "sem controlador aprendido" do DynaKRAG.
* RFPolicy — regressor random forest v̂θ(s,a); escolhe argmax[v̂θ(s,a) − λ·c(s,a)] com λ = 0.

Portão de continuação: os alvos de valor têm escalas diferentes (parar vale SR(s), adquirir
vale ΔSR), então um argmax puro pararia cedo demais. O artigo resolve isso com um modelo de
continuação com limiares permissivos. Aqui começamos com uma REGRA equivalente:
"só pode parar se a última checagem de suficiência disse suficiente" (stop_gate:
sufficient_only). Trocar a regra por um classificador aprendido é um bom passo seguinte.
"""
from __future__ import annotations

import pickle
import random

import numpy as np

from .features import featurize
from .state import ACTION_COST, EvidenceState


def stop_allowed(s: EvidenceState, A: list[str], gate: str) -> bool:
    if A == ["stop_answer"]:
        return True
    if gate == "always":
        return True
    if gate == "never":
        return False
    return s.last_suff == 1  # sufficient_only


class UniformPolicy:
    name = "uniform"

    def __init__(self, seed: int = 13):
        self.seed = seed

    def choose(self, s: EvidenceState, A: list[str]) -> str:
        rng = random.Random(f"{self.seed}-{s.qid}-{len(s.history)}")
        non_stop = [a for a in A if a != "stop_answer"]
        return rng.choice(non_stop) if non_stop else "stop_answer"


class RFPolicy:
    name = "rf"

    def __init__(self, path: str, gate: str = "sufficient_only", lam: float = 0.0):
        with open(path, "rb") as f:
            self.model = pickle.load(f)["model"]
        self.gate, self.lam = gate, lam

    def scores(self, s: EvidenceState, A: list[str]) -> dict[str, float]:
        X = np.stack([featurize(s, a) for a in A])
        v = self.model.predict(X)
        return {a: float(v[i] - self.lam * ACTION_COST[a]) for i, a in enumerate(A)}

    def choose(self, s: EvidenceState, A: list[str]) -> str:
        cand = A if stop_allowed(s, A, self.gate) else [a for a in A if a != "stop_answer"]
        if len(cand) == 1:
            return cand[0]
        sc = self.scores(s, cand)
        return max(cand, key=lambda a: sc[a])


def make_policy(cfg: dict, name: str | None = None):
    name = name or cfg["loop"]["policy"]
    if name == "uniform":
        return UniformPolicy(cfg["data"]["seed"])
    if name == "rf":
        return RFPolicy(cfg["paths"]["rf"], cfg["loop"]["stop_gate"])
    raise ValueError(name)
