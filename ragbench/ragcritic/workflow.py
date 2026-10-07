"""Algoritmo 1 do RAG-Critic ("Generate → Critic → Planning → Execution"), executado POR ESTÁGIO.

    for x in D_test:
        p ← π_α(q, D_q)                         # estágio G: Standard RAG (Prompt A, top-5)
        y ← π_θ(q, D_q, p)                      # estágio C: crítico
        if y == Error:
            p̂ ← π_β(q, D_q, p, y, F, T)          # estágio P: planner escreve o programa
            ŷ ← Executor(p̂)                     # estágio E: executa as 5 funções
    (repetido N vezes — Tabela 6)

Por que por estágio: com 4 GB de VRAM o gerador e o crítico (dois modelos de 3B) não ficam carregados
juntos. Rodando G para todas as perguntas, depois C para todas, depois P/E, o Ollama troca de modelo
poucas vezes em vez de duas vezes por pergunta. Cada estágio grava um JSONL e é retomável.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..config import retrieval_spec
from ..core.datasets import load_corpus, load_records
from ..core.io import read_jsonl, write_jsonl
from ..core.retrieval import load_retriever
from ..core.runner import run_records, score
from ..core.text import doc_contents, extract_final_answer, support_recall
from . import prompts as P
from .critic import Critic
from .executor import Executor
from .planner import plan
from .taxonomy import rule_program

PAPER = "ragcritic"


@dataclass(frozen=True)
class RCVariant:
    name: str
    critic: str = "finetuned"      # finetuned | zeroshot (ablação "w/o Critic Model")
    planning: str = "auto"         # auto (planner) | rule (tabela T, ablação "w/o Auto-Planning")
    iterations: int = 1


def variants(cfg: dict) -> dict[str, RCVariant]:
    return {
        "ragcritic": RCVariant("ragcritic"),
        "ragcritic_n2": RCVariant("ragcritic_n2", iterations=2),
        "ragcritic_zeroshot_critic": RCVariant("ragcritic_zeroshot_critic", critic="zeroshot"),
        "ragcritic_rule_planning": RCVariant("ragcritic_rule_planning", planning="rule"),
    }


class Ctx:
    """Tudo que os estágios de um dataset compartilham."""

    def __init__(self, cfg: dict, dataset: str, llm, limit: int | None = None):
        rc = cfg["ragcritic"]
        self.cfg, self.rc, self.dataset, self.llm = cfg, rc, dataset, llm
        corpus = load_corpus(cfg, dataset)
        self.r = load_retriever(corpus, cfg["paths"]["data"], dataset, retrieval_spec(cfg, PAPER))
        self.records = load_records(cfg, dataset, "eval", int(rc["n_eval"]))
        if limit:
            self.records = self.records[:limit]
        self.by_id = {r["id"]: r for r in self.records}
        self.out = os.path.join(cfg["paths"]["results"], PAPER, dataset)
        self.top_k = int(rc["top_k"])
        self.workers = int(cfg["llm"].get("workers", 1))
        self.default_indent = 16 if str(rc.get("critic_template", "indent20")).endswith("16") else 20
        self.executor = Executor(llm, self.r, max_calls=int(rc["max_program_calls"]), max_topk=int(rc["max_topk"]))

    def path(self, name: str) -> str:
        return os.path.join(self.out, f"{name}.jsonl")

    def run(self, recs: list[dict], fn, name: str, desc: str) -> dict:
        rows = run_records(recs, fn, self.path(name), self.llm, desc=f"[RAG-Critic·{self.dataset}] {desc}",
                           workers=self.workers)
        return {r["id"]: r for r in rows}


# ============================================================================ estágio G — Standard RAG (π_α)
def generate(ctx: Ctx, question: str, contents: list[str], task: str = "rc_generate") -> str:
    prompt = P.PROMPT_A.format(passages=P.passages_block(contents), question=question)
    return ctx.llm.chat("generator", [{"role": "user", "content": prompt}], task=task,
                        max_tokens=int(ctx.rc["generator_max_tokens"]))


def stage_generate(ctx: Ctx) -> dict:
    def fn(rec):
        hits = ctx.r.search(rec["question"], ctx.top_k)
        contents = [doc_contents(h) for h in hits]
        raw = generate(ctx, rec["question"], contents)
        titles = [h["title"] for h in hits]
        return {"pred": extract_final_answer(raw), "raw": raw, "doc_ids": [h["id"] for h in hits],
                "doc_titles": titles, "contents": contents,
                "sr": support_recall(titles, rec.get("supporting_titles") or [])}   # diagnóstico (usa o gold)

    return ctx.run(ctx.records, fn, "standard_rag", "Standard RAG (G)")


# ============================================================================ estágio C — crítico (π_θ)
def critic_key(ctx: Ctx, v: RCVariant) -> str:
    return f"{v.critic}{ctx.default_indent}"


def make_critic(ctx: Ctx, v: RCVariant) -> Critic:
    indent = ctx.default_indent
    mt = int(ctx.rc["critic_max_tokens"])
    if v.critic == "zeroshot":
        return Critic(ctx.llm, "zeroshot", model="generator", indent=indent, max_tokens=mt)
    return Critic(ctx.llm, "finetuned", model="critic", indent=indent, max_tokens=mt)


def stage_critic(ctx: Ctx, v: RCVariant, it: int, ids: list[str], cur: dict) -> dict:
    key = critic_key(ctx, v)
    recs = [ctx.by_id[i] for i in ids]
    critic = make_critic(ctx, v)

    def fn(rec):
        c = cur[rec["id"]]
        out = critic.judge(rec["question"], c["docs"][:5], c["raw"])
        return {**out, "critic": critic.name}

    return ctx.run(recs, fn, f"critic__{key}__it{it}", f"crítico {key} it{it}")


# ============================================================================ estágios P + E
def plan_key(ctx: Ctx, v: RCVariant) -> str:
    return f"{critic_key(ctx, v)}_paper_{v.planning}"


def stage_plan(ctx: Ctx, v: RCVariant, it: int, ids: list[str], cur: dict, crit: dict) -> dict:
    recs = [ctx.by_id[i] for i in ids]
    if v.planning == "rule":
        def fn(rec):
            err, code = rule_program(crit[rec["id"]].get("tag1", []))
            return {"program": code, "rule": err}
    else:
        def fn(rec):
            c = cur[rec["id"]]
            return plan(ctx.llm, rec["question"], c["docs"], c["pred"], crit[rec["id"]],
                        max_tokens=int(ctx.rc["planner_max_tokens"]))
    return ctx.run(recs, fn, f"plan__{plan_key(ctx, v)}__it{it}", f"planner {plan_key(ctx, v)} it{it}")


def stage_execute(ctx: Ctx, v: RCVariant, it: int, ids: list[str], cur: dict, plans: dict) -> dict:
    recs = [ctx.by_id[i] for i in ids]

    def fn(rec):
        c = cur[rec["id"]]
        info = ctx.executor.run(plans[rec["id"]].get("program", ""), rec["question"], c["docs"], c["pred"])
        raw = info.get("answer_raw") or ""
        return {"pred": extract_final_answer(raw) if info["ok"] else "", "raw": raw, "exec_ok": info["ok"],
                "exec_error": info.get("error"), "calls": info.get("calls"), "final_docs": info.get("final_docs"),
                "partial": info.get("partial", False), "queries": info.get("queries"),
                "ret_ids": info.get("ret_ids"), "ret_titles": info.get("ret_titles")}

    return ctx.run(recs, fn, f"exec__{plan_key(ctx, v)}__it{it}", f"executor {plan_key(ctx, v)} it{it}")


# ============================================================================ Algoritmo 1 completo
_USAGE_KEYS = ("llm_calls", "tokens", "llm_seconds")


def run_variant(ctx: Ctx, v: RCVariant) -> list[dict]:
    gen = stage_generate(ctx)
    ids_all = [r["id"] for r in ctx.records if r["id"] in gen]
    cur = {i: {"raw": gen[i]["raw"], "pred": gen[i]["pred"], "docs": gen[i]["contents"][:5], "changed": False,
               "trace": [], "doc_ids": list(gen[i].get("doc_ids") or []),
               "doc_titles": list(gen[i].get("doc_titles") or [])} for i in ids_all}
    usage = {i: {k: gen[i].get(k, 0) or 0 for k in _USAGE_KEYS} for i in ids_all}

    def add_usage(rows: dict):
        for i, r in rows.items():
            for k in _USAGE_KEYS:
                usage[i][k] += r.get(k, 0) or 0

    to_judge = list(ids_all)
    for it in range(1, v.iterations + 1):
        if not to_judge:
            break
        crit = stage_critic(ctx, v, it, to_judge, cur)
        add_usage(crit)
        errs = [i for i in to_judge if crit.get(i, {}).get("judgement") == "Error"]
        for i in to_judge:
            cur[i]["trace"].append({"it": it, "judgement": crit.get(i, {}).get("judgement"),
                                    "tag1": crit.get(i, {}).get("tag1", []), "tag2": crit.get(i, {}).get("tag2", [])})
        changed = []
        if errs:
            plans = stage_plan(ctx, v, it, errs, cur, crit)
            add_usage(plans)
            execs = stage_execute(ctx, v, it, errs, cur, plans)
            add_usage(execs)
            for i in errs:
                e = execs.get(i, {})
                tr = cur[i]["trace"][-1]
                tr.update(program=plans.get(i, {}).get("program", ""), exec_ok=e.get("exec_ok"),
                          exec_error=e.get("exec_error"), prev_pred=cur[i]["pred"], calls=e.get("calls"),
                          queries=e.get("queries"))
                # tudo o que a execução buscou conta como "recuperado" (para saber se a evidência chegou a aparecer)
                for did, title in zip(e.get("ret_ids") or [], e.get("ret_titles") or []):
                    if did not in cur[i]["doc_ids"]:
                        cur[i]["doc_ids"].append(did)
                        cur[i]["doc_titles"].append(title)
                if e.get("exec_ok") and e.get("pred"):
                    fd = e.get("final_docs") or cur[i]["docs"]
                    cur[i].update(raw=e["raw"], pred=e["pred"], docs=list(fd)[:5], changed=True)
                    changed.append(i)
                elif ctx.rc.get("on_exec_fail", "keep_previous") == "empty":
                    cur[i].update(raw="", pred="", changed=True)
                tr["new_pred"] = cur[i]["pred"]
        to_judge = changed  # só respostas que mudaram precisam de novo julgamento
    rows = []
    for i in ids_all:
        rec = ctx.by_id[i]
        tr = cur[i]["trace"]
        rows.append({"id": i, "question": rec["question"], "answers": rec.get("answers", []),
                     "type": rec.get("type", ""), "hops": rec.get("hops"), "pred": cur[i]["pred"], "raw": cur[i]["raw"],
                     "initial_pred": gen[i]["pred"], "changed": cur[i]["changed"],
                     "judgement_it1": tr[0]["judgement"] if tr else None, "trace": tr,
                     "doc_ids": cur[i]["doc_ids"], "doc_titles": cur[i]["doc_titles"], "contents": cur[i]["docs"],
                     "sr": support_recall(cur[i]["doc_titles"], rec.get("supporting_titles") or []),
                     **score(rec, cur[i]["pred"]), **{k: round(usage[i][k], 3) for k in _USAGE_KEYS}})
    write_jsonl(ctx.path(v.name), rows)
    return rows


def run_standard(ctx: Ctx) -> list[dict]:
    gen = stage_generate(ctx)
    return [gen[r["id"]] for r in ctx.records if r["id"] in gen]


def load_method(cfg: dict, dataset: str, method: str) -> list[dict]:
    return read_jsonl(os.path.join(cfg["paths"]["results"], PAPER, dataset, f"{method}.jsonl"))
