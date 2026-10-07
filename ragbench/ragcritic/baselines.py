"""Baselines "critical-based RAG" da Tabela 3 do RAG-Critic, em implementação controlada.

Todos partem da MESMA geração inicial do Standard RAG (Prompt A, top-5 do mesmo retriever) e usam o
mesmo gerador — como o artigo compara tudo com o mesmo backbone.

* Self-Refine (Madaan et al., 2023): feedback do próprio modelo → refino, até ``max_iter`` vezes ou
  até o feedback dizer "NO ISSUES".
* FLARE (Jiang et al., 2023b), versão direta para respostas curtas: se algum token do rascunho tiver
  probabilidade < θ, busca de novo com a pergunta + o rascunho sem os tokens de baixa confiança (β) e
  regenera. Precisa de logprobs (Ollama recente); sem logprobs, sempre faz a busca ativa (θ = 1).
* MetaRAG (Zhou et al., 2024): monitora (compara a resposta com a resposta do conhecimento interno),
  avalia (documentos suficientes? conflito? erro de raciocínio?) e planeja por REGRAS.

Self-RAG não entra: depende de um checkpoint treinado com tokens de reflexão (selfrag_llama2_7b).
"""
from __future__ import annotations

import math

from ..core.jsonutil import parse_obj
from ..core.text import doc_contents, extract_final_answer, f1_score, support_recall
from . import prompts as P
from .workflow import Ctx, generate, stage_generate

_USAGE_KEYS = ("llm_calls", "tokens", "llm_seconds")


def _with_gen_usage(res: dict, g: dict, hits: list[dict] | None = None, rec: dict | None = None) -> dict:
    """Soma o custo da geração inicial (Standard RAG) ao custo do método e registra os documentos vistos
    (os da geração inicial + os de uma nova busca, se houve) — é o que a análise de falhas lê."""
    res["gen_usage"] = {k: g.get(k, 0) for k in _USAGE_KEYS}
    ids, titles = list(g.get("doc_ids") or []), list(g.get("doc_titles") or [])
    for h in hits or []:
        if h["id"] not in ids:
            ids.append(h["id"])
            titles.append(h["title"])
    res["doc_ids"], res["doc_titles"] = ids, titles
    if rec is not None:
        res["sr"] = support_recall(titles, rec.get("supporting_titles") or [])   # diagnóstico (usa o gold)
    return res


# ============================================================================ Self-Refine
def self_refine(ctx: Ctx, rec: dict, g: dict, max_iter: int = 2) -> dict:
    passages = P.passages_block(g["contents"])
    ans_raw, ans = g["raw"], g["pred"]
    feedbacks = []
    for _ in range(max_iter):
        fb = ctx.llm.chat("generator", [{"role": "user", "content": P.SELF_REFINE_FEEDBACK.format(
            passages=passages, question=rec["question"], answer=ans_raw)}], task="selfrefine_feedback", max_tokens=128)
        feedbacks.append(fb)
        if "NO ISSUES" in fb.upper():
            break
        ans_raw = ctx.llm.chat("generator", [{"role": "user", "content": P.SELF_REFINE_REFINE.format(
            passages=passages, question=rec["question"], answer=ans, feedback=fb.strip())}],
            task="selfrefine_refine", max_tokens=int(ctx.rc["generator_max_tokens"]))
        ans = extract_final_answer(ans_raw)
    return _with_gen_usage({"pred": ans, "raw": ans_raw, "feedbacks": feedbacks, "iterations": len(feedbacks)}, g, rec=rec)


# ============================================================================ FLARE
def flare(ctx: Ctx, rec: dict, g: dict, theta: float = 0.4, beta: float = 0.4, max_rounds: int = 1) -> dict:
    prompt = P.PROMPT_A.format(passages=P.passages_block(g["contents"]), question=rec["question"])
    out = ctx.llm.chat_full("generator", [{"role": "user", "content": prompt}], task="flare_draft",
                            max_tokens=int(ctx.rc["generator_max_tokens"]), logprobs=True)
    draft, lp = out["text"], out.get("logprobs")
    rounds, mode = [], "logprobs" if lp else "no-logprobs(always-retrieve)"
    pred, raw = extract_final_answer(draft), draft
    new_hits: list[dict] = []
    for _ in range(max_rounds):
        if lp:
            probs = [(x["token"], math.exp(x["logprob"])) for x in lp]
            low = [t for t, p in probs if p < theta]
            if not low:
                break
            kept = "".join(t for t, p in probs if p >= beta)
        else:
            low, kept = ["*"], draft
        masked = extract_final_answer(kept) or kept
        query = f"{rec['question']} {masked}".strip()
        hits = ctx.r.search(query, ctx.top_k)
        new_hits += hits
        contents = [doc_contents(h) for h in hits]
        raw = generate(ctx, rec["question"], contents, task="flare_regen")
        pred = extract_final_answer(raw)
        rounds.append({"query": query, "n_low": len(low)})
        lp = None  # uma rodada ativa basta para respostas curtas
        break
    return _with_gen_usage({"pred": pred, "raw": raw, "flare_mode": mode, "rounds": rounds}, g, new_hits, rec)


# ============================================================================ MetaRAG
def metarag(ctx: Ctx, rec: dict, g: dict, agree: float = 0.5) -> dict:
    q = rec["question"]
    internal_raw = ctx.llm.chat("generator", [{"role": "user", "content": P.METARAG_INTERNAL.format(question=q)}],
                                task="metarag_internal", max_tokens=96)
    internal = extract_final_answer(internal_raw)
    sim = f1_score(g["pred"], internal)
    info = {"internal": internal, "similarity": round(sim, 3)}
    if sim >= agree:                                              # monitoramento: respostas concordam
        return _with_gen_usage({"pred": g["pred"], "raw": g["raw"], "meta_action": "accept", **info}, g, rec=rec)
    passages = P.passages_block(g["contents"])
    ev_raw = ctx.llm.chat("generator", [{"role": "user", "content": P.METARAG_EVAL.format(
        passages=passages, question=q, answer=g["pred"], internal=internal)}], task="metarag_eval", max_tokens=64,
        fmt=P.METARAG_EVAL_SCHEMA)
    ev = parse_obj(ev_raw, {}) or {}
    ev = ev if isinstance(ev, dict) else {}
    info["evaluation"] = ev
    new_hits: list[dict] = []
    if not ev.get("docs_sufficient", True):                       # planejamento por regra
        nq = ctx.llm.chat("generator", [{"role": "user", "content": P.METARAG_REWRITE.format(question=q)}],
                          task="metarag_rewrite", max_tokens=32)
        nq = next((ln.strip() for ln in nq.splitlines() if ln.strip()), q)
        hits = ctx.r.search(nq, ctx.top_k)
        new_hits = hits[: ctx.top_k - 2]
        info["rewritten_query"] = nq
        contents = [doc_contents(h) for h in hits][: ctx.top_k - 2] + g["contents"][:2]
        raw = generate(ctx, q, contents, task="metarag_regen")
        action = "re-retrieve"
    elif ev.get("conflict") or ev.get("reasoning_error"):
        hint = ("The documents may conflict with your prior knowledge; rely on the documents. "
                if ev.get("conflict") else "Check each reasoning step carefully. ")
        raw = ctx.llm.chat("generator", [{"role": "user", "content": P.METARAG_REASON.format(
            passages=passages, hint=hint, question=q)}], task="metarag_reason", max_tokens=int(ctx.rc["generator_max_tokens"]))
        action = "conflict" if ev.get("conflict") else "reasoning"
    else:
        return _with_gen_usage({"pred": g["pred"], "raw": g["raw"], "meta_action": "keep", **info}, g, rec=rec)
    return _with_gen_usage({"pred": extract_final_answer(raw), "raw": raw, "meta_action": action, **info}, g, new_hits, rec)


BASELINES = {"self_refine": self_refine, "flare": flare, "metarag": metarag}


def run_baseline(ctx: Ctx, name: str) -> list[dict]:
    gen = stage_generate(ctx)
    fn = BASELINES[name]
    kw = (ctx.rc.get("baselines") or {}).get(name, {}) or {}
    rows = ctx.run([r for r in ctx.records if r["id"] in gen], lambda rec: fn(ctx, rec, gen[rec["id"]], **kw),
                   name, name)
    return list(rows.values())
