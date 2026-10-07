"""Treino offline do controlador do DynaKRAG (Eq. 1 + modelo de continuação).

1) **Modelo de valor v̂θ(s, a)** — alvos da Eq. (1):

       v(s, a) = SR(s') − SR(s)     se a é de aquisição (a transição é EXECUTADA a partir de s)
                 1[SR(s) = 1]        se a = sufficiency_check
                 SR(s)               se a = stop_answer

   SR(s) = fração dos documentos de suporte anotados presentes em s. O gold aparece SÓ aqui.
   Os estados vêm de trajetórias de uma política de comportamento uniforme sobre as ações válidas
   (sem parar antes do limite); em CADA estado visitado, TODAS as ações válidas são rotuladas.
   ``stop_answer`` é rotulado em todos os estados (o alvo SR(s) existe sempre), mesmo naqueles em que a
   regra de parada da inferência ainda não o deixaria ser escolhido.

2) **Modelo de continuação** — "trained on trajectory prefixes to estimate the answer utility of
   continuing". Desenrolamos a política de valor com a parada desabilitada até o limite; em cada
   prefixo s_t respondemos com G e comparamos com a resposta no fim da trajetória:
       y_t = 1[F1(resposta em s_T) > F1(resposta em s_t)]      (continuar ajuda)
   O classificador (RF) estima P(continuar ajuda | s_t) com features só observáveis. O limiar τ é calibrado
   simulando a parada SÓ nos prefixos em que a regra de parada a permite (``stop_rule``); com menos de
   ``MIN_STOP_PREFIXES`` prefixos assim não há o que calibrar e vale τ = 0,5.

Ambos: random forest com 300 árvores, folha mínima 8, semente 13 (Tabela 4 do artigo).
"""
from __future__ import annotations

import os
import random
import time

import numpy as np
from tqdm import tqdm

from ..core.io import get_logger, write_json
from ..core.text import best_em_f1
from .controller import LearnedController, controller_path, save_models
from .features import CONT_FEATURE_NAMES, VALUE_FEATURE_NAMES, continuation_features, value_features
from .state import ACQUISITION, STOP, SUFF, control_signature, support_recall
from .validity import stop_allowed, valid_actions

MIN_STOP_PREFIXES = 10      # mínimo de prefixos com parada permitida para calibrar τ
DEFAULT_TAU = 0.5           # sem calibração: só segura a parada se "continuar ajuda" for mais provável que não


# ============================================================================ 1) valor
def collect_value_data(env, records: list[dict], cfg: dict) -> dict:
    dk = cfg["dynakrag"]
    max_steps = int(dk["max_steps"])
    seed = int(cfg.get("seed", 13))
    X, y, acts, groups, state_ids = [], [], [], [], []
    trajectories = []
    sid = 0
    for gi, rec in enumerate(tqdm(records, desc=f"[{env.dataset}] rótulos Eq.1", dynamic_ncols=True)):
        gold = rec.get("supporting_titles") or []
        if not gold:
            continue
        s = env.init_state(rec)
        rng = random.Random(f"{seed}-behavior-{rec['id']}")
        traj = []
        while True:
            A = valid_actions(s, env, max_steps, stop_rule="always")     # rotula stop_answer em todo estado
            sr = support_recall(s.titles(), gold)
            nxt, targets = {}, {}
            for a in A:
                if a in ACQUISITION:
                    s2 = env.step(s, a)
                    nxt[a] = s2
                    targets[a] = support_recall(s2.titles(), gold) - sr
                elif a == SUFF:
                    targets[a] = float(sr >= 1.0 - 1e-9)
                else:
                    targets[a] = sr
            F = value_features(s, A, env, max_steps)
            for i, a in enumerate(A):
                X.append(F[i])
                y.append(targets[a])
                acts.append(a)
                groups.append(gi)
                state_ids.append(sid)
            sid += 1
            non_stop = [a for a in A if a != STOP]
            traj.append({"step": s.steps, "sr": sr, "valid": A, "targets": {a: round(v, 4) for a, v in targets.items()}})
            if not non_stop:
                break
            a = rng.choice(non_stop)
            traj[-1]["chosen"] = a
            s = nxt[a] if a in nxt else env.step(s, a)
        trajectories.append({"id": rec["id"], "steps": traj, "final_sr": support_recall(s.titles(), gold)})
    return {"X": np.asarray(X, dtype=np.float32), "y": np.asarray(y, dtype=np.float32), "action": np.asarray(acts),
            "group": np.asarray(groups), "state": np.asarray(state_ids), "trajectories": trajectories}


def _rf_regressor(cfg: dict):
    from sklearn.ensemble import RandomForestRegressor

    cc = cfg["dynakrag"]["controller"]
    return RandomForestRegressor(n_estimators=int(cc["rf_trees"]), min_samples_leaf=int(cc["rf_min_leaf"]),
                                 random_state=int(cfg.get("seed", 13)), n_jobs=-1)


def _rf_classifier(cfg: dict):
    from sklearn.ensemble import RandomForestClassifier

    cc = cfg["dynakrag"]["controller"]
    return RandomForestClassifier(n_estimators=int(cc["rf_trees"]), min_samples_leaf=int(cc["rf_min_leaf"]),
                                  random_state=int(cfg.get("seed", 13)), n_jobs=-1)


def fit_value_model(data: dict, cfg: dict) -> tuple[object, dict]:
    from sklearn.model_selection import GroupKFold, cross_val_predict

    X, y, acts, groups, states = data["X"], data["y"], data["action"], data["group"], data["state"]
    report: dict = {"n_pairs": int(len(y)), "n_states": int(len(np.unique(states))),
                    "n_questions": int(len(np.unique(groups))), "cv": {}}
    n_splits = min(5, len(np.unique(groups)))
    if n_splits >= 2:
        pred = cross_val_predict(_rf_regressor(cfg), X, y, groups=groups, cv=GroupKFold(n_splits=n_splits))
        for a in sorted(set(acts.tolist())):
            m = acts == a
            report["cv"][a] = {"n": int(m.sum()), "mean_target": float(y[m].mean()),
                               "mae": float(np.abs(pred[m] - y[m]).mean())}
        # concordância do argmax por estado (entre as ações de aquisição/suficiência)
        agree, tot = 0, 0
        for st in np.unique(states):
            m = (states == st) & (acts != STOP)
            if m.sum() < 2:
                continue
            tot += 1
            agree += int(np.argmax(pred[m]) == np.argmax(y[m]))
        report["argmax_agreement"] = agree / tot if tot else None
    model = _rf_regressor(cfg).fit(X, y)
    imp = sorted(zip(VALUE_FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])[:12]
    report["importances"] = [[n, round(float(v), 4)] for n, v in imp]
    return model, report


# ============================================================================ 2) continuação
def collect_continuation_data(env, value_model, records: list[dict], cfg: dict) -> dict:
    dk = cfg["dynakrag"]
    max_steps = int(dk["max_steps"])
    label_kind = dk["controller"].get("continuation_label", "final_better")
    ctrl = LearnedController(value_model, None, lam=float(dk.get("lambda_cost", 0.0)), use_continuation=False,
                             max_steps=max_steps, allow_stop=False)
    X, y, groups, details, f1_now, f1_final, can_stop = [], [], [], [], [], [], []
    rule = getattr(env, "stop_rule", "sufficient")
    for gi, rec in enumerate(tqdm(records, desc=f"[{env.dataset}] rótulos de continuação", dynamic_ncols=True)):
        s = env.init_state(rec)
        prefixes = []
        while True:
            A = valid_actions(s, env, max_steps)
            a, _ = ctrl.choose(s, A, env)
            if a == STOP:
                break
            s = env.step(s, a)
            prefixes.append(s)
        prefixes = [p for p in prefixes if p.docs]
        if len(prefixes) < 2:
            continue
        f1s = []
        for p in prefixes:
            pred, _ = env.answer(p.question, env.passages(p.docs), task="dk_answer")
            f1s.append(best_em_f1(pred, rec.get("answers", []))[1])
        for t, p in enumerate(prefixes[:-1]):
            future = f1s[-1] if label_kind == "final_better" else max(f1s[t + 1:])
            X.append(continuation_features(p, env, max_steps))
            y.append(int(future > f1s[t] + 1e-9))
            groups.append(gi)
            f1_now.append(f1s[t])
            f1_final.append(f1s[-1])
            can_stop.append(int(stop_allowed(p, rule)))
        details.append({"id": rec["id"], "f1_by_prefix": [round(f, 4) for f in f1s],
                        "actions": prefixes[-1].history})
    return {"X": np.asarray(X, dtype=np.float32), "y": np.asarray(y, dtype=np.int64), "group": np.asarray(groups),
            "f1_now": np.asarray(f1_now, dtype=np.float32), "f1_final": np.asarray(f1_final, dtype=np.float32),
            "can_stop": np.asarray(can_stop, dtype=np.int64), "stop_rule": rule, "details": details}


def calibrate_threshold(proba: np.ndarray, data: dict) -> tuple[float, dict]:
    """Escolhe τ simulando a parada nos prefixos de treino: para no 1º prefixo em que a regra de parada
    permite parar (``can_stop``) e P(continuar) < τ.

    Usa probabilidades de validação cruzada (fora da amostra). Em empate, fica o τ MENOR (mais permissivo a
    continuar), no espírito dos "permissive continuation thresholds" do artigo. Com poucos prefixos em que
    parar é permitido, a curva não diz nada: devolve ``DEFAULT_TAU`` e registra por quê.
    """
    groups, f1_now, f1_final = data["group"], data["f1_now"], data["f1_final"]
    can_stop = data.get("can_stop")
    if can_stop is None:
        can_stop = np.ones(len(groups), dtype=np.int64)
    order = {}
    for i, g in enumerate(groups):
        order.setdefault(int(g), []).append(i)
    curve = {}
    for tau in [round(x, 2) for x in np.arange(0.05, 1.0, 0.05)]:
        vals = []
        for idx in order.values():
            f = float(f1_final[idx[0]])
            for i in idx:
                if can_stop[i] and proba[i] < tau:
                    f = float(f1_now[i])
                    break
            vals.append(f)
        curve[tau] = float(np.mean(vals)) if vals else 0.0
    info = {"curve": curve, "never_stop_f1": float(np.mean([f1_final[idx[0]] for idx in order.values()])),
            "n_can_stop": int(np.sum(can_stop)), "n_prefixes": int(len(groups))}
    if info["n_can_stop"] < MIN_STOP_PREFIXES:
        info["note"] = (f"só {info['n_can_stop']} prefixos com parada permitida (mínimo {MIN_STOP_PREFIXES}): "
                        f"τ não calibrado, usando {DEFAULT_TAU}")
        return DEFAULT_TAU, info
    best = max(curve.values())
    tau = min(t for t, v in curve.items() if v >= best - 1e-9)
    return tau, info


def fit_continuation_model(data: dict, cfg: dict) -> tuple[object | None, dict]:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, cross_val_predict

    X, y, groups = data["X"], data["y"], data["group"]
    report: dict = {"n": int(len(y)), "positive_rate": float(y.mean()) if len(y) else None}
    if len(y) < 10 or len(set(y.tolist())) < 2:
        report["note"] = "rótulos insuficientes ou de uma classe só: continuação desativada"
        return None, report
    n_splits = min(5, len(np.unique(groups)))
    if n_splits >= 2:
        try:
            proba = cross_val_predict(_rf_classifier(cfg), X, y, groups=groups, cv=GroupKFold(n_splits=n_splits),
                                      method="predict_proba")[:, 1]
            report["cv_auc"] = float(roc_auc_score(y, proba))
            if "f1_now" in data:
                report["tau"], report["tau_calibration"] = calibrate_threshold(proba, data)
        except ValueError as e:
            report["cv_auc"] = None
            report["cv_note"] = str(e)
    model = _rf_classifier(cfg).fit(X, y)
    imp = sorted(zip(CONT_FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])[:10]
    report["importances"] = [[n, round(float(v), 4)] for n, v in imp]
    return model, report


# ============================================================================ tudo junto
def train_controller(cfg: dict, env, records: list[dict], force: bool = False) -> str:
    log = get_logger()
    path = controller_path(cfg, env.dataset)
    if os.path.exists(path) and not force:
        log.info(f"[{env.dataset}] controlador já treinado: {path}")
        return path
    t0 = time.time()
    vdata = collect_value_data(env, records, cfg)
    if len(vdata["y"]) == 0:
        raise RuntimeError(f"[{env.dataset}] nenhum exemplo de treino com fatos de suporte")
    vmodel, vrep = fit_value_model(vdata, cfg)
    log.info(f"[{env.dataset}] modelo de valor: {vrep['n_pairs']} pares (estado, ação) de {vrep['n_questions']} perguntas; "
             f"concordância do argmax (CV) = {vrep.get('argmax_agreement')}")
    cmodel, crep = None, {"note": "desativado no config"}
    if cfg["dynakrag"]["controller"].get("continuation", True):
        cdata = collect_continuation_data(env, vmodel, records, cfg)
        cmodel, crep = fit_continuation_model(cdata, cfg)
        log.info(f"[{env.dataset}] modelo de continuação: {crep.get('n')} prefixos, "
                 f"taxa de 'continuar ajuda' = {crep.get('positive_rate')}, AUC (CV) = {crep.get('cv_auc')}, "
                 f"τ calibrado = {crep.get('tau')}")
    meta = {"dataset": env.dataset, "n_train": len(records), "control": control_signature(cfg), "value": vrep,
            "continuation": crep, "secs": round(time.time() - t0, 1), "llm": env.llm.summary()}
    save_models(path, vmodel, cmodel, meta)
    write_json(path.replace(".pkl", "_report.json"), {**meta, "trajectories_sample": vdata["trajectories"][:20]})
    log.info(f"[{env.dataset}] controlador salvo em {path} ({meta['secs']}s)")
    return path
