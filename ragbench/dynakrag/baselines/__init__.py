"""Baselines controlados da Tabela 1 do DynaKRAG.

"All baselines use the same backbone, retrieval resources, corpus, and evaluation code as DynaKRAG within
each block." Aqui: mesmo retriever/corpus, mesmo gerador, MESMO prompt de resposta final (``env.answer``)
e mesma extração/avaliação. Os métodos são reimplementações compactas da ideia central de cada artigo
(controladas, sem os modelos treinados originais), exatamente como o DynaKRAG descreve os seus baselines.

| método          | ideia central reproduzida                                                          |
|-----------------|-------------------------------------------------------------------------------------|
| Fixed-K         | top-K estático para a pergunta (grade de K; reporta o melhor)                      |
| IRCoT           | intercala uma frase de CoT e uma busca com essa frase                              |
| Self-Ask+Search | perguntas de acompanhamento respondidas com busca                                  |
| CRAG            | avaliador de retrieval → correct/ambiguous/incorrect → refino ou nova busca         |
| Adaptive-RAG    | classificador de complexidade → sem retrieval / 1 passo / multi-passo (IRCoT)      |
| S2G-RAG         | julgamento estruturado de suficiência → lacunas viram novas queries                |
| CoRAG           | cadeia de sub-queries e sub-respostas (s2 e s3; reporta a mais forte)               |
| PAR²-RAG        | cobertura em largura (sub-perguntas) + refino em profundidade com suficiência      |
"""
from __future__ import annotations

import os
import pickle
import re

from ...core.jsonutil import as_str_list, parse_obj
from ...core.text import best_em_f1, extract_final_answer, first_sentence, norm_title, split_sentences
from .. import prompts as P


# ----------------------------------------------------------------------------- helpers
def _add_unique(docs: list[dict], hits: list[dict], k: int) -> list[dict]:
    ids = {d["id"] for d in docs}
    titles = {norm_title(d["title"]) for d in docs}
    new = []
    for h in hits:
        nt = norm_title(h["title"])
        if h["id"] in ids or nt in titles:
            continue
        new.append(h)
        ids.add(h["id"])
        titles.add(nt)
        if len(new) >= k:
            break
    docs.extend(new)
    return new


def _search_unseen(env, docs: list[dict], query: str, k: int) -> list[dict]:
    hits = env.r.search(query, k=k + 4 * k, exclude_ids={d["id"] for d in docs}, depth=max(60, len(docs) + 8 * k))
    return _add_unique(docs, hits, k)


def _result(pred, raw, docs, n_ret, steps, **extra) -> dict:
    return {"pred": pred, "raw": raw, "n_ret": n_ret, "steps": steps, "doc_titles": [d["title"] for d in docs],
            "doc_ids": [d["id"] for d in docs], **extra}


# ============================================================================ Fixed-K
def fixed_k(env, rec: dict, k: int) -> dict:
    docs = env.r.search(rec["question"], k)
    pred, raw = env.answer(rec["question"], env.passages(docs))
    return _result(pred, raw, docs, 1, 0, k=k)


# ============================================================================ IRCoT (Trivedi et al., 2023)
def ircot(env, rec: dict, k: int = 3, max_steps: int = 4) -> dict:
    q = rec["question"]
    docs: list[dict] = []
    _add_unique(docs, env.r.search(q, k), k)
    n_ret, cot, cot_answer = 1, [], None
    for step in range(max_steps):
        out = env.llm.chat("generator", [
            {"role": "system", "content": P.IRCOT_SYSTEM},
            {"role": "user", "content": P.IRCOT_USER.format(passages=env.passages(docs), question=q,
                                                            cot=" ".join(cot) or "(none)")},
        ], task="ircot_step", max_tokens=64)
        line = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
        line = re.sub(r"^next sentence\s*[:：]\s*", "", line, flags=re.I)
        sent = line if re.search(r"so the answer is", line, re.I) else first_sentence(line)
        if not sent:
            break
        cot.append(sent)
        if re.search(r"so the answer is", sent, re.I):
            cot_answer = extract_final_answer(sent)
            break
        if len(docs) >= env.max_final:
            break
        if not _search_unseen(env, docs, sent, k):
            break
        n_ret += 1
    pred, raw = env.answer(q, env.passages(docs))
    return _result(pred, raw, docs, n_ret, len(cot), cot=cot, cot_answer=cot_answer)


# ============================================================================ Self-Ask + Search (Press et al., 2023)
def self_ask(env, rec: dict, k: int = 3, max_followups: int = 4) -> dict:
    q = rec["question"]
    transcript, docs, n_ret, final = "", [], 0, None
    for i in range(max_followups + 1):
        out = env.llm.chat("generator", [
            {"role": "system", "content": P.SELF_ASK_SYSTEM},
            {"role": "user", "content": P.SELF_ASK_USER.format(question=q, transcript=transcript)},
        ], task="selfask_step", max_tokens=80, stop=["Intermediate answer:", "\nQuestion:"])
        m_final = re.search(r"So the final answer is\s*:\s*(.+)", out, re.I)
        m_follow = re.search(r"Follow up\s*:\s*(.+)", out, re.I)
        if m_final and (not m_follow or m_final.start() < m_follow.start()):
            final = extract_final_answer(m_final.group(0))
            break
        if not m_follow or i == max_followups:
            break
        fq = m_follow.group(1).strip().splitlines()[0].strip()
        hits = _search_unseen(env, docs, fq, k) or env.r.search(fq, k)
        n_ret += 1
        sub, _ = env.answer(fq, env.passages(hits), task="sub_answer")
        transcript += f"Follow up: {fq}\nIntermediate answer: {sub}.\n"
    if final is None:
        out = env.llm.chat("generator", [
            {"role": "system", "content": P.SELF_ASK_SYSTEM},
            {"role": "user", "content": P.SELF_ASK_USER.format(question=q, transcript=transcript) + "So the final answer is:"},
        ], task="selfask_final", max_tokens=32, stop=["\n"])
        final = extract_final_answer("So the final answer is: " + out)
    return _result(final, transcript, docs, n_ret, n_ret, transcript=transcript)


# ============================================================================ CRAG (Yan et al., 2024)
def _crag_refine(env, q: str, docs: list[dict], max_strips: int = 8) -> str:
    strips = []
    for d in docs:
        for s in split_sentences(d["text"]):
            strips.append(f"({d['title']}) {s}")
    strips = strips[:40]
    if not strips:
        return ""
    out = env.llm.chat("generator", [
        {"role": "system", "content": P.CRAG_REFINE_SYSTEM.format(n=max_strips)},
        {"role": "user", "content": P.CRAG_REFINE_USER.format(question=q, strips="\n".join(f"- {s}" for s in strips))},
    ], task="crag_refine", max_tokens=256)
    lines = [ln.strip() for ln in out.splitlines() if ln.strip().startswith(("-", "("))][:max_strips]
    return "\n".join(lines) if lines else "\n".join(f"- {s}" for s in strips[:max_strips])


def crag(env, rec: dict, k: int = 5, max_strips: int = 8) -> dict:
    q = rec["question"]
    hits = env.r.search(q, k)
    out = env.llm.chat("generator", [
        {"role": "system", "content": P.CRAG_EVAL_SYSTEM},
        {"role": "user", "content": P.CRAG_EVAL_USER.format(question=q, passages=env.passages(hits))},
    ], task="crag_eval", max_tokens=96, fmt=P.CRAG_EVAL_SCHEMA)
    obj = parse_obj(out, {}) or {}
    labels = [str(x).lower() for x in (obj.get("labels", []) if isinstance(obj, dict) else [])]
    labels = (labels + ["incorrect"] * len(hits))[: len(hits)]
    if "correct" in labels:
        action = "correct"
    elif all(lb == "incorrect" for lb in labels):
        action = "incorrect"
    else:
        action = "ambiguous"
    parts, docs, n_ret = [], [h for h, lb in zip(hits, labels) if lb != "incorrect"], 1
    if action in ("correct", "ambiguous") and docs:
        parts.append(_crag_refine(env, q, docs, max_strips))
    if action in ("incorrect", "ambiguous"):
        kq = env.llm.chat("generator", [{"role": "system", "content": P.CRAG_REWRITE_SYSTEM},
                                        {"role": "user", "content": P.CRAG_REWRITE_USER.format(question=q)}],
                          task="crag_rewrite", max_tokens=32)
        kq = next((ln.strip() for ln in kq.splitlines() if ln.strip()), q)
        ext = env.r.search(kq, k, exclude_ids={h["id"] for h in hits})
        n_ret += 1
        docs = docs + ext
        parts.append(_crag_refine(env, q, ext, max_strips))
    evidence = "\n".join(p for p in parts if p) or env.passages(hits)
    pred, raw = env.answer(q, evidence)
    return _result(pred, raw, docs or hits, n_ret, n_ret, crag_action=action, labels=labels)


# ============================================================================ Adaptive-RAG (Jeong et al., 2024)
class AdaptiveRouter:
    """Classificador de complexidade treinado com rótulos 'prata' (qual estratégia mais simples acerta)."""

    def __init__(self, pipeline=None, constant: str | None = None, report: dict | None = None):
        self.pipeline, self.constant, self.report = pipeline, constant, report or {}

    def predict(self, question: str) -> str:
        if self.pipeline is None:
            return self.constant or "C"
        return str(self.pipeline.predict([question])[0])


def router_path(cfg: dict, dataset: str) -> str:
    n = cfg["dynakrag"]["baselines"]["adaptive_rag"]["n_train"]
    return os.path.join(cfg["paths"]["data"], dataset, "dynakrag", f"adaptive_router_n{n}.pkl")


def train_adaptive_router(env, records: list[dict], cfg: dict, multihop: bool = True) -> AdaptiveRouter:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    bc = cfg["dynakrag"]["baselines"]
    labels, qs = [], []
    for rec in records:
        ok = {}
        pa, _ = env.answer_no_retrieval(rec["question"])
        ok["A"] = best_em_f1(pa, rec["answers"])[0] == 1.0
        pb = fixed_k(env, rec, bc["adaptive_rag"]["k"])["pred"]
        ok["B"] = best_em_f1(pb, rec["answers"])[0] == 1.0
        pc = ircot(env, rec, **bc["ircot"])["pred"]
        ok["C"] = best_em_f1(pc, rec["answers"])[0] == 1.0
        lab = next((x for x in "ABC" if ok[x]), "C" if multihop else "B")  # viés indutivo do dataset
        labels.append(lab)
        qs.append(rec["question"])
    dist = {x: labels.count(x) for x in "ABC"}
    if len(set(labels)) < 2:
        return AdaptiveRouter(None, labels[0] if labels else "C", {"labels": dist, "note": "uma classe só"})
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
                         LogisticRegression(max_iter=2000, class_weight="balanced", random_state=int(cfg.get("seed", 13))))
    pipe.fit(qs, labels)
    return AdaptiveRouter(pipe, None, {"labels": dist, "n": len(labels)})


def load_or_train_router(env, cfg: dict, records: list[dict], multihop: bool = True) -> AdaptiveRouter:
    path = router_path(cfg, env.dataset)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    router = train_adaptive_router(env, records, cfg, multihop)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(router, f)
    return router


def adaptive_rag(env, rec: dict, router: AdaptiveRouter, k: int = 6, ircot_kw: dict | None = None) -> dict:
    label = router.predict(rec["question"])
    if label == "A":
        pred, raw = env.answer_no_retrieval(rec["question"])
        return _result(pred, raw, [], 0, 0, route="A")
    if label == "B":
        return {**fixed_k(env, rec, k), "route": "B"}
    return {**ircot(env, rec, **(ircot_kw or {})), "route": "C"}


# ============================================================================ S2G-RAG (Li et al., 2026a)
def s2g_rag(env, rec: dict, k: int = 5, gap_k: int = 3, max_rounds: int = 3) -> dict:
    q = rec["question"]
    docs: list[dict] = []
    _add_unique(docs, env.r.search(q, k), k)
    n_ret, rounds, known = 1, [], []
    for _ in range(max_rounds):
        out = env.llm.chat("generator", [
            {"role": "system", "content": P.S2G_SYSTEM},
            {"role": "user", "content": P.S2G_USER.format(question=q, passages=env.passages(docs))},
        ], task="s2g_judge", max_tokens=256, fmt=P.S2G_SCHEMA)
        obj = parse_obj(out, {}) or {}
        obj = obj if isinstance(obj, dict) else {}
        suff = bool(obj.get("sufficient", False))
        gaps = [g for g in (obj.get("gaps") or []) if isinstance(g, dict)][:2]
        known = as_str_list(obj.get("known_facts"))
        rounds.append({"sufficient": suff, "gaps": gaps})
        if suff or not gaps or len(docs) >= env.max_final:
            break
        added = 0
        for g in gaps:
            query = str(g.get("query") or g.get("missing") or "").strip()
            if query:
                added += len(_search_unseen(env, docs, query, gap_k))
                n_ret += 1
        if not added:
            break
    pred, raw = env.answer(q, env.passages(docs))
    return _result(pred, raw, docs, n_ret, len(rounds), rounds=rounds, known_facts=known)


# ============================================================================ CoRAG (Wang et al., 2025)
def corag(env, rec: dict, k: int = 3, steps: int = 3) -> dict:
    q = rec["question"]
    docs: list[dict] = []
    chain: list[tuple[str, str]] = []
    n_ret = 0
    for _ in range(steps):
        chain_txt = "\n".join(f"Sub-question: {a}\nAnswer: {b}" for a, b in chain) or "(none)"
        sq = env.llm.chat("generator", [
            {"role": "system", "content": P.CORAG_SUBQ_SYSTEM},
            {"role": "user", "content": P.CORAG_SUBQ_USER.format(question=q, chain=chain_txt)},
        ], task="corag_subquery", max_tokens=48)
        sq = next((ln.strip() for ln in sq.splitlines() if ln.strip()), q)
        sq = re.sub(r"^(next )?sub-question\s*[:：]\s*", "", sq, flags=re.I)
        hits = _search_unseen(env, docs, sq, k) or env.r.search(sq, k)
        n_ret += 1
        sa, _ = env.answer(sq, env.passages(hits), task="sub_answer")
        chain.append((sq, sa))
    chain_txt = "\n".join(f"Sub-question: {a}\nAnswer: {b}" for a, b in chain)
    evidence = f"Reasoning chain:\n{chain_txt}\n\nDocuments:\n{env.passages(docs)}"
    pred, raw = env.answer(q, evidence)
    return _result(pred, raw, docs, n_ret, steps, chain=chain)


# ============================================================================ PAR²-RAG (Li et al., 2026b)
def par2_rag(env, rec: dict, k: int = 3, max_subq: int = 3, max_refine: int = 2) -> dict:
    from ..env import _first_line

    q = rec["question"]
    out = env.llm.chat("generator", [
        {"role": "system", "content": P.PAR2_DECOMPOSE_SYSTEM.format(n=max_subq)},
        {"role": "user", "content": P.PAR2_DECOMPOSE_USER.format(question=q)},
    ], task="par2_decompose", max_tokens=160, fmt=P.PAR2_DECOMPOSE_SCHEMA)
    subqs = as_str_list(parse_obj(out, {}))[:max_subq] or [q]
    docs: list[dict] = []
    n_ret = 0
    for sq in subqs:                                   # largura: cobertura de todas as sub-perguntas
        _search_unseen(env, docs, sq, k)
        n_ret += 1
    refine = []
    for _ in range(max_refine):                        # profundidade: refino guiado por suficiência
        passages = env.passages(docs)
        j = env.llm.chat("generator", [{"role": "system", "content": P.SUFF_SYSTEM},
                                       {"role": "user", "content": P.SUFF_USER.format(question=q, passages=passages)}],
                         task="par2_suff", max_tokens=24, fmt=P.SUFF_SCHEMA)
        obj = parse_obj(j, {}) or {}
        if isinstance(obj, dict) and obj.get("sufficient"):
            refine.append("sufficient")
            break
        g = env.llm.chat("generator", [{"role": "system", "content": P.GAP_SYSTEM},
                                       {"role": "user", "content": P.GAP_USER.format(question=q, passages=passages)}],
                         task="par2_followup", max_tokens=40)
        gq = _first_line(g, "")
        refine.append(gq)
        if not gq or not _search_unseen(env, docs, gq, k):
            break
        n_ret += 1
    pred, raw = env.answer(q, env.passages(docs))
    return _result(pred, raw, docs, n_ret, len(subqs) + len(refine), subquestions=subqs, refine=refine)
