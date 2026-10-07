"""Estado de evidência s_t do DynaKRAG.

"The state records the question, retrieved documents, retrieval frontier, query and action history,
bridge candidates, and diagnostic feedback." Tudo aqui é observável em inferência: o gold (fatos de
suporte, resposta) nunca entra no estado — só no alvo de treino (Eq. 1).
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field

from ..core.text import norm_title, support_recall

ACQUISITION = ("retrieve_more", "gap_query", "rewrite_query", "bridge_entity_expand")
CONTROL = ("sufficiency_check", "stop_answer")
ACTIONS = ACQUISITION + CONTROL          # as 6 do laço de controle
TERMINAL = "compress_answer_evidence"    # a 7ª: aplicada no término (não é escolhida pelo controlador)
STOP = "stop_answer"
SUFF = "sufficiency_check"

# custo estimado c(s, a): (chamadas de retrieval, chamadas ao LLM). A sonda faz 1 chamada no modo one_step e 1 ou 2
# no two_step; o 1,5 é constante por ação (só pesa na escolha se λ > 0, e as rodadas principais usam λ = 0).
ACTION_COST = {
    "retrieve_more": (1, 0.0),
    "gap_query": (1, 0.0),
    "bridge_entity_expand": (1, 0.0),
    "rewrite_query": (1, 1.0),
    "sufficiency_check": (0, 1.5),
    "stop_answer": (0, 0.0),
}


def scalar_cost(a: str) -> float:
    r, l = ACTION_COST[a]
    return r + l


# ----------------------------------------------------------------------------- regras de controle
# As três regras abaixo mudam a trajetória inteira; por isso ficam gravadas em cada linha de resultado
# (campo ``control``) e no nome do arquivo do controlador treinado. Resultado com outra assinatura é de
# outra versão das regras e não pode ser misturado com os atuais.
LEGACY_CONTROL = "stop=always;probes=inf;probe=two_step"      # regras das versões até 05/10/2026


def control_rules(cfg: dict) -> dict:
    dk = cfg.get("dynakrag") or {}
    cc = dk.get("controller") or {}
    mp = cc.get("max_probes", 1)
    return {"stop_rule": str(cc.get("stop_rule", "sufficient")),
            "max_probes": None if mp in (None, "none", "inf") else int(mp),
            "probe": str(dk.get("probe", "one_step"))}


def control_signature(cfg: dict) -> str:
    r = control_rules(cfg)
    return f"stop={r['stop_rule']};probes={'inf' if r['max_probes'] is None else r['max_probes']};probe={r['probe']}"


def control_slug(cfg: dict) -> str:
    """Sufixo do arquivo do controlador ('' para as regras antigas, que usavam o nome sem sufixo)."""
    if control_signature(cfg) == LEGACY_CONTROL:
        return ""
    r = control_rules(cfg)
    return f"__{r['stop_rule'][:3]}-p{'inf' if r['max_probes'] is None else r['max_probes']}-{r['probe'].split('_')[0]}"


def row_control(row: dict) -> str:
    """Assinatura das regras com que uma linha de resultado foi gerada (linhas antigas não têm o campo)."""
    return str(row.get("control") or LEGACY_CONTROL)


@dataclass
class EvidenceState:
    qid: str
    question: str
    ranking: list = field(default_factory=list)      # [[doc_id, score], ...] ranking inicial (fronteira)
    frontier: int = 0                                 # próxima posição não consumida do ranking inicial
    docs: list = field(default_factory=list)          # [{id, title, text, score, via, step}]
    query: str = ""                                   # query ativa mais recente
    gap: str = ""                                     # descrição da informação ausente (sonda)
    gaps_seen: list = field(default_factory=list)
    suff: int = -1                                    # -1 nunca checado | 0 insuficiente | 1 suficiente
    docs_at_check: int = -1                           # nº de docs quando a sonda rodou pela última vez
    n_checks: int = 0
    bridges_used: list = field(default_factory=list)
    history: list = field(default_factory=list)       # ações executadas (sem stop_answer)
    n_ret: int = 0                                    # ações que produziram retrieval
    n_llm: int = 0                                    # chamadas ao LLM feitas pelas ações

    # ------------------------------------------------------------------
    @property
    def steps(self) -> int:
        return len(self.history)

    def seen_ids(self) -> set:
        return {d["id"] for d in self.docs}

    def seen_titles(self) -> set:
        return {norm_title(d["title"]) for d in self.docs}

    def titles(self) -> list[str]:
        return [d["title"] for d in self.docs]

    def copy(self) -> "EvidenceState":
        new = copy.copy(self)
        for k in ("ranking", "docs", "gaps_seen", "bridges_used", "history"):
            setattr(new, k, list(getattr(self, k)))
        return new

    def to_dict(self, with_text: bool = False) -> dict:
        d = asdict(self)
        d.pop("ranking", None)
        if not with_text:
            d["docs"] = [{k: v for k, v in doc.items() if k != "text"} for doc in self.docs]
        return d


# ``support_recall`` mora em ``core.text`` (a análise de falhas também usa); continua disponível aqui para quem já importava
__all__ = ["ACQUISITION", "CONTROL", "ACTIONS", "TERMINAL", "STOP", "SUFF", "ACTION_COST", "scalar_cost", "EvidenceState",
           "support_recall", "LEGACY_CONTROL", "control_rules", "control_signature", "control_slug", "row_control"]
