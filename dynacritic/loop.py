"""Laço A (DynaKRAG) completo: controle → compressão terminal → resposta. E o baseline de K fixo."""
from __future__ import annotations

import os
import re

from tqdm import tqdm

from . import prompts as P
from .metrics import best_em_f1, support_recall
from .state import EvidenceState
from .utils import numbered_passages, read_jsonl, write_jsonl
from .validity import valid_actions

_FINAL = re.compile(r"so the final answer is\s*:?\s*(.*)", re.I | re.S)


def extract_answer(raw: str) -> str:
    m = _FINAL.search(raw or "")
    ans = (m.group(1) if m else (raw or "").strip().splitlines()[-1] if raw and raw.strip() else "")
    ans = ans.strip().splitlines()[0] if ans.strip() else ""
    return ans.strip().strip("*").strip().rstrip(".").strip()


def run_episode(env, policy, s: EvidenceState, max_final_docs: int, trace: list | None = None) -> EvidenceState:
    while True:
        A = valid_actions(s, max_final_docs)
        a = policy.choose(s, A)
        if trace is not None:
            trace.append({"step": len(s.history), "valid": A, "chosen": a})
        if a == "stop_answer":
            return s
        s = env.step(s, a)


def compress(env, s: EvidenceState, docs: list[dict]) -> str:
    lc = env.cfg["loop"]
    out = env.llm.chat("generator", [
        {"role": "system", "content": P.COMPRESS_SYSTEM.format(n=lc["compress_max_snippets"])},
        {"role": "user", "content": P.COMPRESS_USER.format(question=s.question, passages=numbered_passages(docs))},
    ], task="compress_answer_evidence", max_tokens=lc["compress_max_tokens"])
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return "\n".join(lines[: lc["compress_max_snippets"]]) or numbered_passages(docs)


def generate_answer(env, question: str, evidence: str, tips: str | None = None) -> tuple[str, str]:
    system = P.ANSWER_SYSTEM + (P.ANSWER_TIPS.format(tips=tips) if tips else "")
    raw = env.llm.chat("generator", [
        {"role": "system", "content": system},
        {"role": "user", "content": P.ANSWER_USER.format(evidence=evidence, question=question)},
    ], task="generate_answer", max_tokens=env.cfg["loop"]["answer_max_tokens"] + 32)
    return extract_answer(raw), raw


def final_context(env, s: EvidenceState, use_compression: bool) -> tuple[list[dict], str]:
    docs = s.docs[: env.cfg["loop"]["max_final_docs"]]
    if use_compression and docs:
        return docs, compress(env, s, docs)
    return docs, numbered_passages(docs)


def _row(rec, s, pred, raw, evidence, usage, extra=None):
    em, f1 = best_em_f1(pred, rec["answers"])
    row = {"qid": rec["qid"], "question": rec["question"], "answers": rec["answers"], "pred": pred, "raw": raw,
           "em": em, "f1": f1, "sr": support_recall(s.titles(), rec["supporting_titles"]),
           "n_ret": s.n_ret, "steps": len(s.history), "history": s.history, "evidence": evidence,
           "usage": usage, "state": s.to_dict()}
    row.update(extra or {})
    return row


def run_loop_a(cfg: dict, env, policy, split: str, name: str, use_compression: bool | None = None,
               limit: int | None = None) -> str:
    use_compression = cfg["loop"]["compression"] if use_compression is None else use_compression
    recs = read_jsonl(cfg["paths"][split])[:limit]
    rows = []
    for rec in tqdm(recs, desc=f"laço A [{name}]"):
        with env.llm.track() as u:
            trace = []
            s = env.init_state(rec["qid"], rec["question"])
            s = run_episode(env, policy, s, cfg["loop"]["max_final_docs"], trace)
            _, evidence = final_context(env, s, use_compression)
            pred, raw = generate_answer(env, s.question, evidence)
        rows.append(_row(rec, s, pred, raw, evidence, dict(u), {"trace": trace}))
    out = os.path.join(cfg["paths"]["runs"], name, "preds.jsonl")
    write_jsonl(out, rows)
    return out


def run_fixed_k(cfg: dict, env, split: str, k: int, name: str, limit: int | None = None) -> str:
    recs = read_jsonl(cfg["paths"][split])[:limit]
    rows = []
    for rec in tqdm(recs, desc=f"fixed-K={k}"):
        with env.llm.track() as u:
            s = EvidenceState(qid=rec["qid"], question=rec["question"], query=rec["question"])
            for h in env.r.search(rec["question"], k):
                s.docs.append({"idx": h["idx"], "title": h["title"], "text": h["text"], "score": h["score"],
                               "via": "fixed_k"})
            s.n_ret = 1
            evidence = numbered_passages(s.docs)
            pred, raw = generate_answer(env, s.question, evidence)
        rows.append(_row(rec, s, pred, raw, evidence, dict(u)))
    out = os.path.join(cfg["paths"]["runs"], name, "preds.jsonl")
    write_jsonl(out, rows)
    return out
