"""Etapas 1–3 da construção do sistema de erros (ver ``__init__``)."""
from __future__ import annotations

import csv
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

from ...config import retrieval_spec
from ...core.datasets import is_prepared, load_corpus, load_records
from ...core.io import ensure_dir, get_logger, read_jsonl, write_json, write_jsonl
from ...core.jsonutil import parse_obj
from ...core.retrieval import load_retriever
from ...core.runner import run_records
from ...core.text import doc_contents, extract_final_answer
from ...core.tokens import count_tokens
from .. import prompts as P
from ..critic import normalize_judgement
from ..taxonomy import TIER2, TIER2_PARENT


def es_dir(cfg: dict) -> str:
    return ensure_dir(os.path.join(cfg["paths"]["results"], "ragcritic", "errorsystem"))


# ============================================================================ etapa 1
def stage1(cfg: dict, llm) -> list[dict]:
    """Amostragem (pool de modelos, Prompt A) + julgamento com análise detalhada (Prompt B)."""
    log = get_logger()
    es = cfg["ragcritic"]["errorsystem"]
    out_dir = es_dir(cfg)
    items = []
    for ds in es["datasets"]:
        if not is_prepared(cfg, ds):
            log.warning(f"[sistema de erros] {ds} não preparado — pulando")
            continue
        recs = load_records(cfg, ds, "train") or load_records(cfg, ds, "eval")
        recs = recs[-int(es["n_per_dataset"]):]  # o fim do pool de treino: disjunto das avaliações
        r = load_retriever(load_corpus(cfg, ds), cfg["paths"]["data"], ds, retrieval_spec(cfg, "ragcritic"))
        for rec in recs:
            hits = r.search(rec["question"], int(cfg["ragcritic"]["top_k"]))
            items.append({**rec, "dataset": ds, "contents": [doc_contents(h) for h in hits]})
    responses = []
    for model in es["pool"]:  # modelo por modelo: o Ollama carrega cada um uma vez só
        def fn(rec, model=model):
            prompt = P.PROMPT_A.format(passages=P.passages_block(rec["contents"]), question=rec["question"])
            raw = llm.chat(model, [{"role": "user", "content": prompt}], task="es_sample",
                           max_tokens=int(cfg["ragcritic"]["generator_max_tokens"]), temperature=float(es["temperature"]))
            return {"pred": extract_final_answer(raw), "raw": raw, "model": model, "dataset": rec["dataset"]}
        mrecs = [{**it, "id": f"{it['dataset']}::{it['id']}"} for it in items]
        rows = run_records(mrecs, fn, os.path.join(out_dir, "stage1_responses", f"{_slug(model)}.jsonl"), llm,
                           desc=f"[sistema de erros] amostragem {model}")
        by = {x["id"]: x for x in mrecs}
        for row in rows:
            responses.append({**by[row["id"]], "model": model, "response": row.get("raw", "")})

    def judge(rec):
        rag_input = P.PROMPT_A.format(passages=P.passages_block(rec["contents"]), question=rec["question"])
        raw = llm.chat("judge", [{"role": "user", "content": P.PROMPT_B.format(
            rag_input=rag_input, prediction=rec["response"], golden=" | ".join(rec["answers"]))}],
            task="es_judge", max_tokens=384)
        obj = parse_obj(raw, {}) or {}
        obj = obj if isinstance(obj, dict) else {}
        j = normalize_judgement(obj.get("Judgement", obj.get("judgement", "")))
        return {"judgement": j, "analysis": str(obj.get("Error analysis") or obj.get("Error_analysis") or ""),
                "judge_raw": raw}

    jrecs = [{**x, "id": f"{x['id']}::{_slug(x['model'])}"} for x in responses]
    jrows = run_records(jrecs, judge, os.path.join(out_dir, "stage1_judged.jsonl"), llm, desc="[sistema de erros] julgamento (Prompt B)")
    by = {x["id"]: x for x in jrecs}
    merged = [{**by[r["id"]], "judgement": r["judgement"], "analysis": r["analysis"]} for r in jrows]
    write_jsonl(os.path.join(out_dir, "D_error_candidates.jsonl"), merged)
    log.info(f"[sistema de erros] {len(merged)} respostas julgadas; "
             f"{sum(m['judgement'] == 'Error' for m in merged)} com erro")
    return merged


def _slug(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()


# ============================================================================ etapa 2
def stage2(cfg: dict, llm, judged: list[dict]) -> tuple[list[dict], dict]:
    es = cfg["ragcritic"]["errorsystem"]
    errors = [x for x in judged if x["judgement"] == "Error" and x.get("analysis")]

    def tag(rec):
        raw = llm.chat("judge", [{"role": "user", "content": P.PROMPT_C.format(analysis=rec["analysis"])}],
                       task="es_tag", max_tokens=256)
        obj = parse_obj(raw, None)
        if isinstance(obj, dict):
            obj = [obj]
        tags = []
        if isinstance(obj, list):
            for t in obj:
                if isinstance(t, dict) and str(t.get("tag", "")).strip():
                    tags.append(str(t["tag"]).strip())
        return {"tags": tags, "json_ok": isinstance(obj, list), "tag_raw": raw}

    rows = run_records(errors, tag, os.path.join(es_dir(cfg), "stage2_tags.jsonl"), llm,
                       desc="[sistema de erros] tags open-set (Prompt C)")
    by = {x["id"]: x for x in errors}
    tagged = [{**by[r["id"]], "atomic": r["tags"]} for r in rows if r.get("json_ok") and r.get("tags")]
    # normalização (artigo): remove rótulos com frequência < α e com mais de 25 tokens
    freq = Counter(t for x in tagged for t in x["atomic"])
    alpha, max_tok = int(es["alpha"]), int(es["max_label_tokens"])
    keep = {t for t, c in freq.items() if c >= alpha and count_tokens(t) <= max_tok}
    for x in tagged:
        x["atomic"] = [t for t in x["atomic"] if t in keep]
    tagged = [x for x in tagged if x["atomic"]]
    stats = {"n_errors": len(errors), "n_tagged": len(tagged), "n_raw_labels": len(freq), "n_kept_labels": len(keep),
             "alpha": alpha, "top_labels": freq.most_common(40)}
    write_json(os.path.join(es_dir(cfg), "stage2_label_stats.json"), stats)
    return tagged, {t: freq[t] for t in keep}


# ============================================================================ etapa 3
def embed_labels(labels: list[str], model: str) -> np.ndarray:
    from ...core.retrieval import get_encoder

    enc = get_encoder(model, "cpu", 128)
    return np.asarray(enc.encode(labels, batch_size=64, convert_to_numpy=True), dtype=np.float32)


def stage3(cfg: dict, llm, tagged: list[dict], label_freq: dict) -> dict:
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler

    log = get_logger()
    es = cfg["ragcritic"]["errorsystem"]
    labels = sorted(label_freq)
    if len(labels) < 2:
        raise RuntimeError("poucos rótulos depois da normalização — aumente n_per_dataset ou reduza alpha")
    E = embed_labels(labels, es["embed_model"])
    Z = StandardScaler().fit_transform(E)
    k = min(int(es["n_clusters"]), len(labels))
    assign = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    rng = random.Random(int(cfg.get("seed", 13)))
    clusters = []
    for c in range(k):
        members = [labels[i] for i in range(len(labels)) if assign[i] == c]
        sample = rng.sample(members, min(len(members), int(es["labels_per_cluster"])))
        name = llm.chat("summarizer", [{"role": "user", "content": (
            "These are fine-grained error labels from retrieval-augmented generation outputs:\n"
            + "\n".join(f"- {m}" for m in sample)
            + "\n\nSummarize the central error type they share as a short category name (2-6 words). "
              "Reply with the name only.")}], task="es_cluster_name", max_tokens=24)
        name = (name.strip().splitlines() or [f"Cluster {c}"])[0].strip().strip('"')
        clusters.append({"cluster": c, "tier2_name": name, "size": len(members),
                         "frequency": int(sum(label_freq[m] for m in members)), "labels": members})
    # sugestão automática de nível 1: nível 2 oficial mais próximo (a revisar por humanos)
    off = embed_labels(TIER2, es["embed_model"])
    names = embed_labels([c["tier2_name"] for c in clusters], es["embed_model"])

    def unit(x):
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)

    sim = unit(names) @ unit(off).T
    for i, c in enumerate(clusters):
        j = int(np.argmax(sim[i]))
        c["closest_official_tier2"] = TIER2[j]
        c["similarity"] = round(float(sim[i, j]), 3)
        c["suggested_tier1"] = TIER2_PARENT[TIER2[j]]
    out_dir = es_dir(cfg)
    write_json(os.path.join(out_dir, "taxonomy.json"), clusters)
    with open(os.path.join(out_dir, "tier1_manual_review.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "tier2_name", "size", "suggested_tier1", "closest_official_tier2", "tier1_final (edite)"])
        for c in clusters:
            w.writerow([c["cluster"], c["tier2_name"], c["size"], c["suggested_tier1"], c["closest_official_tier2"],
                        c["suggested_tier1"]])
    # mapeamento reverso: cada amostra recebe tags dos 3 níveis (Eq. 1)
    lab2c = {labels[i]: int(assign[i]) for i in range(len(labels))}
    data, dist = [], Counter()
    for x in tagged:
        cs = sorted({lab2c[t] for t in x["atomic"] if t in lab2c})
        t2 = [clusters[c]["tier2_name"] for c in cs]
        t1 = sorted({clusters[c]["suggested_tier1"] for c in cs})
        dist.update(t1)
        data.append({"id": x["id"], "dataset": x["dataset"], "model": x["model"], "question": x["question"],
                     "contents": x["contents"], "prediction": x["response"], "judgement": "Error",
                     "tag1": t1, "tag2": t2, "tag3": x["atomic"], "analysis": x["analysis"]})
    write_jsonl(os.path.join(out_dir, "D_error_qa.jsonl"), data)
    summary = {"n_labels": len(labels), "n_clusters": k, "n_samples": len(data),
               "tier1_distribution": dict(dist.most_common())}
    write_json(os.path.join(out_dir, "summary.json"), summary)
    log.info(f"[sistema de erros] {k} clusters, {len(data)} amostras em D_Error^QA → {out_dir}")
    return summary


def run_all(cfg: dict, llm) -> dict:
    judged = stage1(cfg, llm)
    tagged, freq = stage2(cfg, llm, judged)
    summary = stage3(cfg, llm, tagged, freq)
    # também guarda as respostas corretas (para balancear o SFT, como no artigo)
    correct = [x for x in judged if x["judgement"] == "Correct"]
    write_jsonl(os.path.join(es_dir(cfg), "D_correct.jsonl"),
                [{"id": x["id"], "dataset": x["dataset"], "model": x["model"], "question": x["question"],
                  "contents": x["contents"], "prediction": x["response"], "judgement": "Correct"} for x in correct])
    by_model = defaultdict(lambda: [0, 0])
    for x in judged:
        by_model[x["model"]][0] += x["judgement"] == "Error"
        by_model[x["model"]][1] += 1
    summary["error_rate_by_model"] = {m: round(a / b, 3) for m, (a, b) in by_model.items() if b}
    write_json(os.path.join(es_dir(cfg), "summary.json"), summary)
    return summary


__all__ = ["stage1", "stage2", "stage3", "run_all", "read_jsonl"]
