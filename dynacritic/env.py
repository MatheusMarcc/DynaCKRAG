"""Ambiente do laço A: executa as operações atômicas do DynaKRAG sobre o estado.

Cada ação recebe um estado e devolve um NOVO estado (sem efeitos colaterais) — isso é o que
permite a rotulagem offline: a partir do mesmo estado, testamos todas as ações válidas.
"""
from __future__ import annotations

import re
from collections import Counter

import json_repair

from . import prompts as P
from .state import EvidenceState
from .utils import norm_title, numbered_passages

_CAP = re.compile(r"\b[A-Z][\w'’\-]*(?:\s+(?:of|the|de|da|do|del|von|van|and|&)?\s*[A-Z][\w'’\-]*)*")
_BAD = {"The", "A", "An", "In", "On", "He", "She", "It", "They", "His", "Her", "This", "That", "After",
        "Before", "During", "As", "At", "By", "For", "From", "Its", "Their", "There", "When", "While"}


def bridge_candidates(s: EvidenceState, limit: int = 5) -> list[str]:
    """Entidades-ponte: nomes próprios citados nos docs que ainda não têm documento próprio.

    O artigo não especifica o detector; esta heurística funciona bem no HotpotQA, onde a
    entidade-ponte costuma ser o título do parágrafo do 2º salto, mencionado no 1º.
    """
    q = norm_title(s.question)
    have = {norm_title(t) for t in s.titles()} | {norm_title(b) for b in s.bridges_used}
    cnt, first = Counter(), {}
    for d in s.docs:
        for m in _CAP.finditer(d["text"]):
            e = m.group(0).strip(" -'")
            if e in _BAD or len(e) < 4:
                continue
            ne = norm_title(e)
            if ne in have or ne in q:
                continue
            cnt[e] += 1
            first.setdefault(e, len(first))
    ranked = sorted(cnt, key=lambda e: (-(cnt[e] + (" " in e)), first[e]))
    return ranked[:limit]


class Env:
    def __init__(self, cfg: dict, retriever, llm):
        self.cfg = cfg
        self.r = retriever
        self.llm = llm
        self.k = cfg["retriever"]["docs_per_action"]
        self.depth = cfg["retriever"]["initial_depth"]

    # ------------------------------------------------------------ inicialização
    def init_state(self, qid: str, question: str) -> EvidenceState:
        hits = self.r.search(question, self.depth)
        lc = self.cfg["loop"]
        return EvidenceState(qid=qid, question=question, query=question,
                             ranking=[[h["idx"], h["score"]] for h in hits],
                             step_budget=lc["max_steps"], ret_budget=lc["max_retrievals"])

    # ------------------------------------------------------------ helpers
    def _add(self, s: EvidenceState, hits: list[dict], via: str) -> None:
        seen = s.seen_idx()
        seen_t = {norm_title(t) for t in s.titles()}
        added = 0
        for h in hits:
            if h["idx"] in seen or norm_title(h["title"]) in seen_t:
                continue
            s.docs.append({"idx": h["idx"], "title": h["title"], "text": h["text"],
                           "score": float(h.get("score", 0.0)), "via": via})
            seen.add(h["idx"]); seen_t.add(norm_title(h["title"]))
            added += 1
            if added >= self.k:
                break
        s.n_ret += 1

    def _search_unseen(self, query: str) -> list[dict]:
        return self.r.search(query, self.k + len(self._cur_seen) + 5)

    def passages(self, s: EvidenceState, limit: int | None = None) -> str:
        return numbered_passages(s.docs[:limit] if limit else s.docs)

    # ------------------------------------------------------------ transição
    def step(self, s: EvidenceState, action: str) -> EvidenceState:
        s2 = s.copy()
        self._cur_seen = s2.seen_idx()
        getattr(self, action)(s2)
        if action != "stop_answer":
            s2.history.append(action)
        return s2

    # ------------------------------------------------------------ operações atômicas
    def retrieve_more(self, s: EvidenceState) -> None:
        """Avança a fronteira do ranking inicial (sem chamar o LLM)."""
        seen = s.seen_idx()
        docs = self.r.docs
        batch = []
        while s.frontier < len(s.ranking) and len(batch) < self.k:
            idx, score = s.ranking[s.frontier]
            s.frontier += 1
            if idx not in seen:
                batch.append(dict(docs[idx], score=score))
        self._add(s, batch, "retrieve_more")

    def gap_query(self, s: EvidenceState) -> None:
        """Busca com a descrição da informação ausente produzida pela sonda (sem LLM)."""
        self._add(s, self._search_unseen(s.gap), "gap_query")
        s.query, s.gap_used = s.gap, True

    def rewrite_query(self, s: EvidenceState) -> None:
        q = self.llm.chat("generator", [
            {"role": "system", "content": P.REWRITE_SYSTEM},
            {"role": "user", "content": P.REWRITE_USER.format(question=s.question, query=s.query,
                                                              passages=self.passages(s))},
        ], task="rewrite_query", max_tokens=48)
        q = (q.strip().splitlines() or [s.question])[0].strip().strip('"') or s.question
        self._add(s, self._search_unseen(q), "rewrite_query")
        s.query = q

    def bridge_entity_expand(self, s: EvidenceState) -> None:
        cands = bridge_candidates(s)
        if not cands:
            return
        e = cands[0]
        self._add(s, self._search_unseen(e), "bridge_entity_expand")
        s.bridges_used.append(e)

    def sufficiency_check(self, s: EvidenceState) -> None:
        out = self.llm.chat("generator", [
            {"role": "system", "content": P.SUFFICIENCY_SYSTEM},
            {"role": "user", "content": P.SUFFICIENCY_USER.format(question=s.question, passages=self.passages(s))},
        ], task="sufficiency_check", max_tokens=96, fmt=P.SUFFICIENCY_SCHEMA)
        j = json_repair.loads(out) if out else {}
        j = j if isinstance(j, dict) else {}
        ok = bool(j.get("sufficient", False))
        s.last_suff = int(ok)
        s.docs_at_last_check = len(s.docs)
        missing = str(j.get("missing", "") or "").strip()
        if not ok and missing:
            s.gap, s.gap_used = missing, False

    def stop_answer(self, s: EvidenceState) -> None:
        pass
