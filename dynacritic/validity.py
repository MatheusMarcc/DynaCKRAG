"""Camada rígida de validade A(s): regras determinísticas aplicadas ANTES da ordenação aprendida.

Correspondem à Tabela 5 do DynaKRAG. O modelo aprendido só escolhe entre ações que fazem
sentido; as regras garantem que nenhuma transição indefinida ou redundante seja executada.
"""
from __future__ import annotations

from .env import bridge_candidates
from .state import EvidenceState


def valid_actions(s: EvidenceState, max_final_docs: int = 12) -> list[str]:
    A = ["stop_answer"]
    if len(s.history) >= s.step_budget:
        return A
    can_ret = s.n_ret < s.ret_budget and len(s.docs) < max_final_docs
    seen = s.seen_idx()
    if can_ret and any(idx not in seen for idx, _ in s.ranking[s.frontier:]):
        A.append("retrieve_more")
    if s.docs:
        if can_ret:
            A.append("rewrite_query")
        if s.docs_at_last_check != len(s.docs):          # sonda não redundante
            A.append("sufficiency_check")
    if can_ret and s.gap and not s.gap_used:
        A.append("gap_query")
    if can_ret and bridge_candidates(s):
        A.append("bridge_entity_expand")
    return A
