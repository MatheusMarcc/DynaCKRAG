"""Prepara os dados: amostra perguntas do HotpotQA (distractor) e monta um corpus reduzido.

Por que corpus reduzido? Indexar a Wikipedia inteira (21M passagens) não cabe num notebook.
Aqui o corpus é a UNIÃO dos parágrafos (2 de suporte + 8 distratores) de todas as perguntas
amostradas de treino e de avaliação. Cada pergunta passa a competir com milhares de parágrafos
de outras perguntas — um "open-domain em miniatura". Os números absolutos não batem com o
fullwiki do artigo, mas as comparações relativas (uniforme vs RF, com/sem crítico) valem.
"""
from __future__ import annotations

import json
import random

from .utils import norm_title, write_jsonl


# ------------------------------------------------------------------ conversores
def _from_hf(ex: dict) -> dict:
    ctx = ex["context"]
    paragraphs = [
        {"title": t, "text": " ".join(s.strip() for s in sents).strip()}
        for t, sents in zip(ctx["title"], ctx["sentences"])
    ]
    sup = list(dict.fromkeys(ex["supporting_facts"]["title"]))
    return {
        "qid": ex["id"], "question": ex["question"], "answers": [ex["answer"]],
        "type": ex.get("type", ""), "level": ex.get("level", ""),
        "paragraphs": paragraphs, "supporting_titles": sup,
    }


def _from_official(ex: dict) -> dict:
    paragraphs = [{"title": t, "text": " ".join(s.strip() for s in sents).strip()} for t, sents in ex["context"]]
    sup = list(dict.fromkeys(t for t, _ in ex["supporting_facts"]))
    return {
        "qid": ex["_id"], "question": ex["question"], "answers": [ex["answer"]],
        "type": ex.get("type", ""), "level": ex.get("level", ""),
        "paragraphs": paragraphs, "supporting_titles": sup,
    }


# ------------------------------------------------------------------ carregadores
def load_hf(split: str, n: int, seed: int) -> list[dict]:
    """split: 'train' | 'validation'. Usa streaming para não baixar os ~600 MB do treino."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split, streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out = []
    for ex in ds:
        out.append(_from_hf(ex))
        if len(out) >= n:
            break
    return out


def load_official_json(path: str, n: int, seed: int) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rng = random.Random(seed)
    rng.shuffle(raw)
    return [_from_official(ex) for ex in raw[:n]]


# ------------------------------------------------------------------ corpus
def build_corpus(*record_lists: list[dict]) -> list[dict]:
    seen, corpus = {}, []
    for recs in record_lists:
        for r in recs:
            for p in r["paragraphs"]:
                key = norm_title(p["title"])
                if key in seen or not p["text"]:
                    continue
                seen[key] = len(corpus)
                corpus.append({"idx": len(corpus), "title": p["title"], "text": p["text"]})
    return corpus


def prepare(cfg: dict, source: str = "hf", train_json: str | None = None, dev_json: str | None = None) -> None:
    n_tr, n_dev, seed = cfg["data"]["n_train"], cfg["data"]["n_dev"], cfg["data"]["seed"]
    if source == "hf":
        train = load_hf("train", n_tr, seed)
        dev = load_hf("validation", n_dev, seed)
    else:
        train = load_official_json(train_json, n_tr, seed)
        dev = load_official_json(dev_json, n_dev, seed)
    # os parágrafos ficam no corpus; os registros guardam só o necessário
    corpus = build_corpus(train, dev)
    strip = lambda rs: [{k: v for k, v in r.items() if k != "paragraphs"} for r in rs]  # noqa: E731
    write_jsonl(cfg["paths"]["train"], strip(train))
    write_jsonl(cfg["paths"]["dev"], strip(dev))
    write_jsonl(cfg["paths"]["corpus"], corpus)
    print(f"train={len(train)}  dev={len(dev)}  corpus={len(corpus)} parágrafos")
