"""Estado de evidências compartilhado pelos dois laços (DynaKRAG e RAG-Critic)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

ACQUISITION = ["retrieve_more", "gap_query", "rewrite_query", "bridge_entity_expand"]
CONTROL = ["sufficiency_check", "stop_answer"]
ACTIONS = ACQUISITION + CONTROL
# custo estimado por ação (1 = só retrieval, 2 = chamada ao LLM + retrieval)
ACTION_COST = {"retrieve_more": 1, "gap_query": 1, "bridge_entity_expand": 1, "rewrite_query": 2,
               "sufficiency_check": 1, "stop_answer": 0}


@dataclass
class EvidenceState:
    qid: str
    question: str
    ranking: list = field(default_factory=list)   # [[idx, score], ...] ranking inicial (fronteira)
    frontier: int = 0                              # próxima posição não consumida do ranking
    docs: list = field(default_factory=list)       # [{"idx","title","text","score","via"}]
    query: str = ""                                # query ativa
    gap: str = ""                                  # descrição da informação ausente
    gap_used: bool = False
    last_suff: int = -1                            # -1 nunca checado | 0 insuficiente | 1 suficiente
    docs_at_last_check: int = -1
    bridges_used: list = field(default_factory=list)
    history: list = field(default_factory=list)    # ações executadas (sem stop)
    n_ret: int = 0                                 # ações que produziram retrieval
    step_budget: int = 4
    ret_budget: int = 4

    # ------------------------------------------------------------------
    def seen_idx(self) -> set:
        return {d["idx"] for d in self.docs}

    def titles(self) -> list[str]:
        return [d["title"] for d in self.docs]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "EvidenceState":
        return cls(**d)

    def copy(self) -> "EvidenceState":
        return EvidenceState.from_dict(
            {k: (list(v) if isinstance(v, list) else v) for k, v in self.to_dict().items()}
        )
