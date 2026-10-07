"""Features observáveis do par (estado, ação) e do estado (modelo de continuação).

Apêndice do artigo, "Runtime features": "The action-value model observes only quantities available at
inference: question length and shape, number of retrieved documents and distinct titles, frontier
position, retrieval-score statistics, lexical question–evidence overlap, detected bridge-entity count,
current action identity, prior action counts, and estimated action cost. The continuation model
additionally uses evidence-burden and trajectory-history features. Gold answers, supporting facts,
support recall, and answer-quality scores are excluded."
"""
from __future__ import annotations

import re

import numpy as np

from ..core.text import content_tokens, norm_title
from .state import ACTIONS, ACTION_COST, EvidenceState

_WH = ["who", "what", "when", "where", "which", "how", "why"]
_CMP = {"or", "both", "first", "earlier", "later", "older", "younger", "more", "less", "same", "than", "compare",
        "longer", "shorter", "larger", "smaller", "higher", "lower", "before", "after", "either", "neither"}
_YESNO = {"is", "are", "was", "were", "did", "does", "do", "can", "could", "has", "have", "had", "will", "would"}
_CAPTOK = re.compile(r"\b[A-Z][a-z]\w*")

QUESTION_FEATURES = ["q_len", "q_caps", "q_digit", "q_cmp", "q_yesno"] + [f"q_wh_{w}" for w in _WH] + ["q_wh_other"]
EVIDENCE_FEATURES = ["n_docs", "n_titles", "ev_words", "frontier", "score_max", "score_mean", "score_min",
                     "next_frontier_score", "frontier_top_score", "frontier_drop", "overlap", "max_doc_overlap",
                     "q_ent_in_titles", "n_bridges"]
STATE_FEATURES = ["steps", "steps_left", "n_ret", "ret_left", "has_gap", "suff_none", "suff_no", "suff_yes",
                  "docs_since_check", "n_checks", "n_llm"] + [f"cnt_{a}" for a in ACTIONS]
ACTION_FEATURES = [f"act_{a}" for a in ACTIONS] + ["cost_ret", "cost_llm"]

VALUE_FEATURE_NAMES = QUESTION_FEATURES + EVIDENCE_FEATURES + STATE_FEATURES + ACTION_FEATURES
CONT_FEATURE_NAMES = QUESTION_FEATURES + EVIDENCE_FEATURES + STATE_FEATURES


def question_features(q: str) -> list[float]:
    toks = q.lower().replace("?", " ").split()
    first = toks[0] if toks else ""
    wh = [0.0] * (len(_WH) + 1)
    hit = next((i for i, w in enumerate(_WH) if w in toks[:3]), None)
    wh[hit if hit is not None else len(_WH)] = 1.0
    return [float(len(toks)), float(len(_CAPTOK.findall(q))), float(bool(re.search(r"\d", q))),
            float(bool(_CMP & set(toks))), float(first in _YESNO)] + wh


def _evidence_features(s: EvidenceState, env) -> list[float]:
    qt = set(content_tokens(s.question))
    scores = [d.get("score", 0.0) for d in s.docs] or [0.0]
    ev_tokens = set()
    max_doc = 0.0
    words = 0
    for d in s.docs:
        toks = set(content_tokens(d["title"] + " " + d["text"]))
        ev_tokens |= toks
        words += len(d["text"].split())
        if qt:
            max_doc = max(max_doc, len(qt & toks) / len(qt))
    overlap = len(qt & ev_tokens) / len(qt) if qt else 0.0
    nxt = s.ranking[s.frontier][1] if s.frontier < len(s.ranking) else 0.0
    top = s.ranking[0][1] if s.ranking else 0.0
    q_ents = [norm_title(e) for e in re.findall(r"\b[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*)*", s.question)]
    titles = {norm_title(d["title"]) for d in s.docs}
    ent_hit = (sum(any(e in t or t in e for t in titles) for e in q_ents) / len(q_ents)) if q_ents else 0.0
    n_bridges = len(env.bridge_candidates(s)) if s.docs else 0
    return [float(len(s.docs)), float(len(titles)), float(words), float(s.frontier), float(max(scores)),
            float(np.mean(scores)), float(min(scores)), float(nxt), float(top), float(nxt / top) if top else 0.0,
            overlap, max_doc, ent_hit, float(n_bridges)]


def _state_features(s: EvidenceState, max_steps: int, retrieval_cap: int | None) -> list[float]:
    ret_left = (retrieval_cap - s.n_ret) if retrieval_cap is not None else (max_steps - s.steps)
    return ([float(s.steps), float(max_steps - s.steps), float(s.n_ret), float(ret_left), float(bool(s.gap)),
             float(s.suff == -1), float(s.suff == 0), float(s.suff == 1),
             float(len(s.docs) - max(0, s.docs_at_check)) if s.docs_at_check >= 0 else float(len(s.docs)),
             float(s.n_checks), float(s.n_llm)]
            + [float(s.history.count(a)) for a in ACTIONS])


def state_vector(s: EvidenceState, env, max_steps: int, retrieval_cap: int | None = None) -> list[float]:
    return question_features(s.question) + _evidence_features(s, env) + _state_features(s, max_steps, retrieval_cap)


def action_vector(a: str) -> list[float]:
    r, l = ACTION_COST[a]
    return [float(a == x) for x in ACTIONS] + [float(r), float(l)]


def value_features(s: EvidenceState, actions: list[str], env, max_steps: int, retrieval_cap: int | None = None) -> np.ndarray:
    base = state_vector(s, env, max_steps, retrieval_cap)
    return np.asarray([base + action_vector(a) for a in actions], dtype=np.float32)


def continuation_features(s: EvidenceState, env, max_steps: int, retrieval_cap: int | None = None) -> np.ndarray:
    return np.asarray(state_vector(s, env, max_steps, retrieval_cap), dtype=np.float32)
