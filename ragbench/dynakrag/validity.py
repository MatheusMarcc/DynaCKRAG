"""Camada rígida de validade A(s_t) — Tabela 5 do artigo, aplicada ANTES da ordenação aprendida.

"The validity layer first removes actions that cannot be executed; the value model then chooses among
the remaining operations." O modelo aprendido só escolhe entre ações que fazem sentido; as regras
impedem transições indefinidas, esgotadas ou redundantes.

| ação                  | condição principal (Tabela 5)                                         |
|-----------------------|-----------------------------------------------------------------------|
| retrieve_more         | a fronteira e o orçamento de retrieval ainda não acabaram             |
| gap_query             | há uma lacuna não vazia registrada (e ela ainda traz docs não vistos)  |
| rewrite_query         | já existe evidência recuperada                                        |
| bridge_entity_expand  | existe ao menos um candidato a entidade-ponte                          |
| sufficiency_check     | existe evidência e a sonda não é redundante no estado atual            |
| stop_answer           | "enabled by the stopping and continuation logic or by a hard cap"      |

Limites: no máximo ``max_steps`` passos de aquisição/controle; com ``retrieval_cap``, as ações de
retrieval são removidas quando o orçamento de chamadas de retrieval se esgota.

Duas regras que o artigo não escreve por extenso, mas que as tabelas dele fixam (ver docs/DynaKRAG.md):

* ``stop_rule`` — quando ``stop_answer`` entra em A(s_t):
    - ``sufficient`` (padrão): só depois que a sonda disse "suficiente". Sem a sonda, nenhum exemplo do artigo
      escolhe parar (Tabela 8: 4,000 passos; "none selects stop_answer"); com ela, só param cedo os 26% / 20% /
      5% que a Tabela 6 mostra saindo depois de uma única busca.
    - ``probe``: depois de qualquer veredito da sonda (quem decide é só o modelo de continuação).
    - ``always``: sempre válido — a regra das versões anteriores deste código, mantida para reproduzir
      resultados antigos.
* ``max_probes`` — a sonda roda no máximo esse número de vezes por trajetória. Padrão 1: a Tabela 7 conta
  0,977–0,999 sondas por pergunta nos três datasets, nunca mais de uma. ``None`` = sem limite.

Quando nenhuma outra ação é executável, ``stop_answer`` é a única saída, qualquer que seja a regra.
"""
from __future__ import annotations

from .state import STOP, SUFF, EvidenceState

STOP_RULES = ("sufficient", "probe", "always")
_UNSET = object()


def stop_allowed(s: EvidenceState, stop_rule: str) -> bool:
    if stop_rule == "always":
        return True
    if stop_rule == "probe":
        return s.suff != -1
    if stop_rule == "sufficient":
        return s.suff == 1
    raise ValueError(f"stop_rule desconhecida: {stop_rule!r} (opções: {', '.join(STOP_RULES)})")


def valid_actions(s: EvidenceState, env, max_steps: int, retrieval_cap: int | None = None,
                  disabled: frozenset | set = frozenset(), stop_rule: str | None = None,
                  max_probes=_UNSET) -> list[str]:
    """Ações executáveis em ``s``. ``stop_rule`` e ``max_probes`` vêm do ambiente quando não são passados."""
    if stop_rule is None:
        stop_rule = getattr(env, "stop_rule", "sufficient")
    if max_probes is _UNSET:
        max_probes = getattr(env, "max_probes", 1)
    if s.steps >= max_steps:
        return [STOP]
    A: list[str] = []
    budget_ok = retrieval_cap is None or s.n_ret < retrieval_cap
    if budget_ok:
        if env.frontier_has_unseen(s):
            A.append("retrieve_more")
        if s.gap and env.query_has_unseen(s, s.gap):
            A.append("gap_query")
        if s.docs:
            A.append("rewrite_query")
        if s.docs and env.bridge_candidates(s):
            A.append("bridge_entity_expand")
    if s.docs and s.docs_at_check != len(s.docs) and (max_probes is None or s.n_checks < int(max_probes)):
        A.append(SUFF)
    A = [a for a in A if a not in disabled]
    if not A or stop_allowed(s, stop_rule):
        A.append(STOP)
    return A


def retrieval_exhausted(s: EvidenceState, retrieval_cap: int | None) -> bool:
    return retrieval_cap is not None and s.n_ret >= retrieval_cap
