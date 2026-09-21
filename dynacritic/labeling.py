"""Rotulagem offline + treino do controlador (Equação 1 do DynaKRAG).

Para cada pergunta de treino, uma política de comportamento (uniforme sobre ações válidas,
sem parar antes do limite) visita estados. Em CADA estado visitado, todas as ações válidas
são rotuladas:

    v(s,a) = SR(s') − SR(s)      se a é de aquisição   (executa a transição)
             1[SR(s) = 1]         se a = sufficiency_check
             SR(s)                se a = stop_answer

SR = fração dos títulos de suporte presentes no estado. O gold aparece SÓ aqui, no alvo.
Custo: só rewrite_query chama o LLM durante a ramificação (gap/bridge/retrieve_more são
retrieval puro); o cache torna a segunda passada gratuita.
"""
from __future__ import annotations

import pickle
import random

import numpy as np
from tqdm import tqdm

from .features import FEATURE_NAMES, featurize
from .metrics import support_recall
from .state import ACQUISITION
from .utils import read_jsonl
from .validity import valid_actions


def label_offline(cfg: dict, env, limit: int | None = None) -> dict:
    recs = read_jsonl(cfg["paths"]["train"])[:limit]
    X, y, acts, groups = [], [], [], []
    seed = cfg["data"]["seed"]
    max_docs = cfg["loop"]["max_final_docs"]
    for gi, rec in enumerate(tqdm(recs, desc="rotulagem ΔSR")):
        gold = rec["supporting_titles"]
        s = env.init_state(rec["qid"], rec["question"])
        rng = random.Random(f"{seed}-{rec['qid']}")
        while True:
            A = valid_actions(s, max_docs)
            sr = support_recall(s.titles(), gold)
            nxt = {}
            for a in A:
                if a in ACQUISITION:
                    s2 = env.step(s, a)
                    nxt[a] = s2
                    target = support_recall(s2.titles(), gold) - sr
                elif a == "sufficiency_check":
                    target = float(sr == 1.0)
                else:  # stop_answer
                    target = sr
                X.append(featurize(s, a)); y.append(target); acts.append(a); groups.append(gi)
            non_stop = [a for a in A if a != "stop_answer"]
            if not non_stop:
                break
            a = rng.choice(non_stop)
            s = nxt[a] if a in nxt else env.step(s, a)
    data = {"X": np.stack(X), "y": np.asarray(y, np.float32), "action": np.asarray(acts),
            "group": np.asarray(groups)}
    np.savez_compressed(cfg["paths"]["labels"], **data)
    print(f"{len(y)} pares (estado, ação) rotulados a partir de {len(recs)} perguntas")
    return data


def train_controller(cfg: dict) -> None:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict

    d = np.load(cfg["paths"]["labels"], allow_pickle=False)
    X, y, acts, groups = d["X"], d["y"], d["action"], d["group"]
    lc = cfg["loop"]
    make = lambda: RandomForestRegressor(n_estimators=lc["rf_trees"], min_samples_leaf=lc["rf_min_leaf"],  # noqa: E731
                                         random_state=cfg["data"]["seed"], n_jobs=-1)
    # validação cruzada AGRUPADA por pergunta (estados da mesma pergunta não vazam entre folds)
    n_splits = min(5, len(np.unique(groups)))
    if n_splits >= 2:
        pred = cross_val_predict(make(), X, y, groups=groups, cv=GroupKFold(n_splits=n_splits))
        print("MAE por ação (validação cruzada agrupada por pergunta):")
        for a in sorted(set(acts)):
            m = acts == a
            print(f"  {a:22s} n={m.sum():5d}  alvo médio={y[m].mean():.3f}  MAE={np.abs(pred[m] - y[m]).mean():.3f}")
    model = make().fit(X, y)
    imp = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda t: -t[1])[:10]
    print("features mais importantes:", ", ".join(f"{n}={v:.3f}" for n, v in imp))
    with open(cfg["paths"]["rf"], "wb") as f:
        pickle.dump({"model": model, "features": FEATURE_NAMES}, f)
    print(f"controlador salvo em {cfg['paths']['rf']}")
