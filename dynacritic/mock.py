"""LLM de mentira, determinístico, para testar o pipeline sem GPU nem Ollama (comando smoke).

As respostas imitam o formato de cada tarefa. A qualidade não importa: o objetivo é exercitar
todo o encanamento (estado, validade, rotulagem, RF, crítico, reparo, métricas).
"""
from __future__ import annotations

import json
import re

_CAP = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")


def _user(messages):
    return next((m["content"] for m in messages if m["role"] == "user"), "")


def _system(messages):
    return next((m["content"] for m in messages if m["role"] == "system"), "")


def _question(text: str) -> str:
    m = re.search(r"Question:\s*(.+)", text)
    return m.group(1).strip() if m else ""


def _entities(text: str) -> list[str]:
    return [e for e in _CAP.findall(text) if e not in {"Question", "Passages", "Evidence", "Document", "Snippets",
                                                       "Previous", "New", "Critic", "So", "In", "The", "Model"}]


def _country(text: str) -> str | None:
    m = re.findall(r"\b([A-Z][a-z]+(?:ia|land|stan))\b", text)
    return m[0] if m else None


class MockBrain:
    def respond(self, task: str, messages: list[dict]) -> str:
        u, s = _user(messages), _system(messages)
        q = _question(u) or _question(s)
        if task == "sufficiency_check":
            n = len(re.findall(r"^\[\d+\]", u, re.M))
            ents = [e for e in _entities(u) if e not in q]
            return json.dumps({"sufficient": n >= 6 and bool(_country(u)),
                               "missing": f"{ents[-1]} country" if ents else q})
        if task in ("rewrite_query", "repair_rewrite", "gap_from_critique"):
            ents = [e for e in _entities(u) if e not in q]
            return (ents[0] + " located in country") if ents else (q or u[:80])
        if task == "compress_answer_evidence":
            lines = re.findall(r"^\[\d+\]\s*(.+)$", u, re.M)
            return "\n".join("- " + ln.split(". ")[0] for ln in lines[:8])
        if task == "generate_answer":
            c = _country(u.split("Question:")[0])
            ents = _entities(u.split("Question:")[0])
            return f"Reasoning omitted.\nSo the final answer is: {c or (ents[0] if ents else 'unknown')}"
        if task in ("critic", "critic_bench"):
            pred = s.split("Model's Prediction:")[-1][:300]
            qq = _question(s)
            if _country(pred) and sum(map(ord, qq)) % 3 != 0:
                return '{"Judgement": "Correct"}'
            return json.dumps({"Judgement": "Error",
                               "Error_analysis": "The passages do not contain the country of the birthplace; "
                                                 "the second hop is missing.",
                               "tag1": ["Incomplete Information"],
                               "tag2": ["Insufficient or Incomplete Information Retrieval"],
                               "tag3": ["Missing Bridge Entity"]})
        if task == "planner_code":
            return ("```python\n"
                    "queries = RewriteQuery(question, \"clarify\")\n"
                    "docs = Retrieval(queries[0], 3) + doc_list\n"
                    "final_answer = GenerateAnswer(question, docs, \"Give only the country name.\")\n```")
        if task == "planner_menu":
            ents = [e for e in _entities(u) if e not in q]
            return json.dumps({"steps": [{"action": "retrieve_new", "argument": (ents[0] if ents else q) + " country"},
                                         {"action": "drop_docs", "argument": "9"}],
                               "answer_instruction": "Give only the country name."})
        if task == "repair_decompose":
            return json.dumps({"subquestions": [f"Where was the person in '{u[:40]}' born?",
                                                "Which country is that city in?"]})
        if task == "repair_refine":
            doc = u.split("Document:")[-1].strip()
            return doc[:200]
        return "ok"
