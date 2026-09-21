"""Features do par (estado, ação) — SÓ quantidades observáveis em inferência.

Nada de resposta gold, fatos de suporte ou recall de suporte aqui: isso vazaria o rótulo.
"""
from __future__ import annotations

import numpy as np

from .env import bridge_candidates
from .state import ACTION_COST, ACTIONS, EvidenceState
from .utils import content_tokens

_WH = ["who", "what", "when", "where", "which", "how", "is/are/did"]
_CMP = {"both", "same", "or", "older", "younger", "first", "earlier", "later", "more", "larger",
        "longer", "higher", "compare", "than"}

FEATURE_NAMES = (
    ["q_len", "q_is_comparison"] + [f"q_wh_{w}" for w in _WH]
    + ["n_docs", "n_titles", "frontier", "n_ret", "step", "steps_left", "ret_left",
       "score_max", "score_mean", "score_min", "next_frontier_score", "score_drop",
       "overlap", "n_bridges", "has_gap", "gap_used", "last_suff", "docs_since_check"]
    + [f"act_{a}" for a in ACTIONS] + [f"prev_{a}" for a in ACTIONS] + ["act_cost"]
)


def _wh(q: str) -> list[float]:
    first = q.lower().split()[:1]
    w = first[0] if first else ""
    v = [0.0] * len(_WH)
    if w in _WH[:-1]:
        v[_WH.index(w)] = 1.0
    else:
        v[-1] = 1.0
    return v


def featurize(s: EvidenceState, action: str) -> np.ndarray:
    q = s.question
    qt = set(content_tokens(q))
    scores = [d["score"] for d in s.docs] or [0.0]
    ev = set()
    for d in s.docs:
        ev.update(content_tokens(d["title"] + " " + d["text"]))
    overlap = len(qt & ev) / max(1, len(qt))
    nxt = s.ranking[s.frontier][1] if s.frontier < len(s.ranking) else 0.0
    top = s.ranking[0][1] if s.ranking else 1.0
    prev = [float(s.history.count(a)) for a in ACTIONS]
    f = (
        [len(q.split()), float(bool(qt & _CMP))] + _wh(q)
        + [len(s.docs), len({d["title"] for d in s.docs}), s.frontier, s.n_ret, len(s.history),
           s.step_budget - len(s.history), s.ret_budget - s.n_ret,
           max(scores), float(np.mean(scores)), min(scores), nxt, (nxt / top) if top else 0.0,
           overlap, len(bridge_candidates(s)), float(bool(s.gap)), float(s.gap_used), float(s.last_suff),
           float(len(s.docs) - max(0, s.docs_at_last_check))]
        + [float(action == a) for a in ACTIONS] + prev + [float(ACTION_COST[action])]
    )
    return np.asarray(f, dtype=np.float32)
