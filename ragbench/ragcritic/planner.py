"""Planner π_β: recebe (q, D_q, p, y, F, T) e escreve um programa Python com as 5 funções (Eq. 2).

O planner recebe y completo (tags de nível 1 e 2 e a análise do crítico) e a tabela erro→ação T, como na
Eq. (2) do artigo.
"""
from __future__ import annotations

from . import prompts as P
from .taxonomy import mapping_table_text


def planner_messages(question: str, doc_list: list[str], previous_pred: str, critique: dict | None) -> list[dict]:
    critique = critique or {}
    tag2 = critique.get("tag2") or []
    error_type = ",".join(tag2) if tag2 else ("Unknown error" if critique else "No Error")
    extra = ""
    if critique:
        extra = P.PLANNER_EXTRA_PAPER.format(tier1=", ".join(critique.get("tag1") or []) or "unknown",
                                             analysis=(critique.get("analysis") or "not provided").strip(),
                                             table=mapping_table_text(critique.get("tag1") or []))
    user = P.PLANNER_USER.format(question=question, doc_list=str(doc_list), previous_pred=previous_pred,
                                 error_type=error_type, extra=extra)
    return [{"role": "system", "content": P.PLANNER_SYSTEM}, {"role": "user", "content": user}]


def extract_code(agent_output: str) -> str:
    """Mesma regra do executor oficial: o último bloco ```python (ou o texto todo se não houver cerca)."""
    text = agent_output or ""
    if "```python" in text:
        return text.split("```python")[-1].split("```")[0].strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[-2].strip()
    return text.strip()


def plan(llm, question: str, doc_list: list[str], previous_pred: str, critique: dict | None,
         max_tokens: int = 768, model: str = "planner") -> dict:
    msgs = planner_messages(question, doc_list, previous_pred, critique)
    raw = llm.chat(model, msgs, task="rc_plan", max_tokens=max_tokens)
    return {"plan_raw": raw, "program": extract_code(raw)}
