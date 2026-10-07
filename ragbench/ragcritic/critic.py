"""Crítico π_θ: julga a predição e devolve tags de erro em 3 níveis.

Dois modos:
  * ``finetuned`` — RAG-Critic-3B (pesos oficiais em GGUF). Prompt D exatamente como no treino, enviado
    como mensagem de USUÁRIO com o system padrão do Qwen (igual ao ``demo.ipynb`` oficial), via ChatML cru.
  * ``zeroshot`` — um LLM qualquer com o Prompt E (inclui as listas de tags de nível 1 e 2), como os
    baselines da Tabela 5.
"""
from __future__ import annotations

import re

from ..core.jsonutil import as_str_list, parse_obj
from . import prompts as P

_JUDGE_RE = re.compile(r"""['"]?Judg[e]?ment['"]?\s*[:=]\s*['"]?([A-Za-z]+)""", re.I)


def normalize_judgement(j) -> str:
    s = str(j or "").strip().lower()
    if s.startswith("correct") or s in ("right", "true", "yes"):
        return "Correct"
    if any(w in s for w in ("error", "incorrect", "wrong", "false", "partially")):
        return "Error"
    return "Unknown"


def parse_critic_output(text: str) -> dict:
    obj = parse_obj(text, None)
    if isinstance(obj, list):
        obj = next((x for x in obj if isinstance(x, dict)), None)
    ok = isinstance(obj, dict)
    obj = obj if ok else {}
    judg = obj.get("Judgement", obj.get("judgement", obj.get("Judgment", obj.get("judgment"))))
    if judg is None:
        m = _JUDGE_RE.search(text or "")
        judg = m.group(1) if m else ""
    analysis = (obj.get("Error_analysis") or obj.get("Error analysis") or obj.get("error_analysis")
                or obj.get("analysis") or "")
    return {"judgement": normalize_judgement(judg), "analysis": str(analysis) if analysis != "None" else "",
            "tag1": as_str_list(obj.get("tag1")), "tag2": as_str_list(obj.get("tag2")),
            "tag3": as_str_list(obj.get("tag3")), "parse_ok": ok}


class Critic:
    def __init__(self, llm, mode: str = "finetuned", model: str = "critic", indent: int = 20, max_tokens: int = 768):
        if mode not in ("finetuned", "zeroshot"):
            raise ValueError(mode)
        self.llm, self.mode, self.model, self.indent, self.max_tokens = llm, mode, model, indent, max_tokens

    @property
    def name(self) -> str:
        return f"{self.mode}:{self.llm.resolve(self.model)}:indent{self.indent}"

    def prompt(self, question: str, passages: list[str], prediction: str) -> str:
        kind = "critic" if self.mode == "finetuned" else "baseline"
        return P.render_critic(question, passages[:5], prediction, self.indent, kind)

    def run_prompt(self, prompt: str, task: str) -> dict:
        if self.mode == "finetuned":
            raw = self.llm.chat(self.model, [{"role": "user", "content": prompt}], task=task,
                                max_tokens=self.max_tokens, template="chatml")
        else:
            raw = self.llm.chat(self.model, [{"role": "user", "content": prompt}], task=task,
                                max_tokens=self.max_tokens)
        out = parse_critic_output(raw)
        out["raw"] = raw
        return out

    def judge(self, question: str, passages: list[str], prediction: str, task: str | None = None) -> dict:
        task = task or ("rc_critic" if self.mode == "finetuned" else "rc_critic_zeroshot")
        return self.run_prompt(self.prompt(question, passages, prediction), task)
