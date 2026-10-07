"""Datasets → registros unificados + corpus reduzido por dataset.

Registro unificado (um por pergunta)::

    {"id", "dataset", "question", "answers": [...], "supporting_titles": [...], "type", "hops"}

Corpus reduzido
---------------
O pacote não indexa a Wikipedia inteira (5M+ parágrafos). O corpus de cada dataset é a
UNIÃO dos parágrafos (suporte + distratores) de três grupos de perguntas: avaliação, treino e um
grupo "extra" usado só para acrescentar distratores. Cada pergunta passa a competir com milhares
de parágrafos de outras perguntas — um open-domain em miniatura. Os números absolutos ficam mais
altos que no fullwiki dos artigos; as comparações entre métodos continuam justas porque todos usam
o mesmo corpus, o mesmo retriever e o mesmo gerador.

Fontes (tentadas em ordem; a primeira que funcionar vence)
---------------------------------------------------------
* arquivo local no formato oficial, se configurado em ``data.local.<dataset>``;
* Hugging Face (streaming, sem baixar o dataset inteiro);
* download direto dos arquivos do repositório no Hugging Face (fallback).
"""
from __future__ import annotations

import gzip
import json
import os
import random
import time
from typing import Callable, Iterable

from .io import ensure_dir, get_logger, read_json, read_jsonl, write_json, write_jsonl
from .text import norm_title

# ----------------------------------------------------------------------------- fontes
HF_SOURCES = {
    "hotpotqa": [
        {"repo": "hotpotqa/hotpot_qa", "config": "distractor", "train": "train", "eval": "validation"},
    ],
    "2wiki": [
        {"repo": "framolfese/2WikiMultihopQA", "config": None, "train": "train", "eval": "validation"},
        {"repo": "xanhho/2WikiMultihopQA", "config": None, "train": "train", "eval": "dev"},
        {"repo": "xanhho/2WikiMultihopQA", "config": None, "train": "train", "eval": "validation"},
    ],
    "musique": [
        {"repo": "dgslibisey/MuSiQue", "config": None, "train": "train", "eval": "validation"},
        {"repo": "bdsaglam/musique", "config": "answerable", "train": "train", "eval": "validation"},
        {"repo": "bdsaglam/musique", "config": None, "train": "train", "eval": "validation"},
    ],
    "nq": [
        {"repo": "Tevatron/wikipedia-nq", "config": None, "train": "train", "eval": "dev"},
        {"repo": "florin-hf/nq_open_gold", "config": None, "train": "train", "eval": "test", "conv": "gold_passage"},
        {"repo": "florin-hf/nq_open_gold", "config": None, "train": "train", "eval": "validation", "conv": "gold_passage"},
    ],
    "triviaqa": [
        {"repo": "Tevatron/wikipedia-trivia", "config": None, "train": "train", "eval": "dev"},
        {"repo": "mandarjoshi/trivia_qa", "config": "rc.wikipedia", "train": "train", "eval": "validation",
         "conv": "trivia_rc"},
    ],
}

DATASET_NAMES = {"hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue", "nq": "NQ",
                 "triviaqa": "TriviaQA", "toy_multihop": "Toy multi-hop", "toy_single": "Toy single-hop"}


def _dedup(xs: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(x for x in xs if x))


# ----------------------------------------------------------------------------- conversores
def _ctx_pairs(ctx) -> list[tuple[str, list[str]]]:
    """Aceita contexto como lista de [título, [frases]] (JSON oficial) ou dict de listas (HF)."""
    if isinstance(ctx, dict):
        titles = ctx.get("title") or ctx.get("titles") or []
        sents = ctx.get("sentences") or ctx.get("content") or ctx.get("contents") or []
        return list(zip(titles, sents))
    out = []
    for c in ctx or []:
        if isinstance(c, dict):
            out.append((c.get("title", ""), c.get("sentences") or c.get("content") or []))
        else:
            out.append((c[0], c[1]))
    return out


def _sf_titles(sf) -> list[str]:
    if isinstance(sf, dict):
        return _dedup(sf.get("title") or [])
    return _dedup(x[0] if isinstance(x, (list, tuple)) else x.get("title", "") for x in sf or [])


def conv_hotpot_like(ex: dict, dataset: str) -> dict:
    paragraphs = []
    for t, sents in _ctx_pairs(ex.get("context")):
        if isinstance(sents, str):
            text = sents
        else:
            text = " ".join(s.strip() for s in sents).strip()
        paragraphs.append({"title": t, "text": text})
    ans = ex.get("answer")
    answers = ans if isinstance(ans, list) else [ans]
    return {
        "id": str(ex.get("id") or ex.get("_id")), "dataset": dataset, "question": ex["question"].strip(),
        "answers": [a for a in answers if a], "supporting_titles": _sf_titles(ex.get("supporting_facts")),
        "type": ex.get("type", "") or "", "level": ex.get("level", "") or "", "hops": None,
        "paragraphs": paragraphs,
    }


def conv_musique(ex: dict, dataset: str = "musique") -> dict | None:
    if ex.get("answerable") is False:
        return None
    paras = ex.get("paragraphs") or []
    paragraphs = [{"title": p.get("title", ""), "text": p.get("paragraph_text") or p.get("text", "")} for p in paras]
    sup = _dedup(p.get("title", "") for p in paras if p.get("is_supporting"))
    qid = str(ex.get("id", ""))
    try:
        hops = int(qid.split("hop")[0]) if "hop" in qid else len(ex.get("question_decomposition") or [])
    except ValueError:
        hops = len(ex.get("question_decomposition") or [])
    answers = _dedup([ex.get("answer", "")] + list(ex.get("answer_aliases") or []))
    return {"id": qid, "dataset": dataset, "question": ex["question"].strip(), "answers": answers,
            "supporting_titles": sup, "type": f"{hops}hop" if hops else "", "level": "", "hops": hops or None,
            "paragraphs": paragraphs}


def conv_tevatron(ex: dict, dataset: str, max_neg: int = 20) -> dict | None:
    q = ex.get("query") or ex.get("question")
    answers = ex.get("answers") or ex.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    pos = ex.get("positive_passages") or []
    neg = (ex.get("negative_passages") or [])[:max_neg]
    if not q or not answers:
        return None
    paragraphs = [{"title": p.get("title", ""), "text": p.get("text", "")} for p in list(pos) + list(neg)]
    return {"id": str(ex.get("query_id") or ex.get("id") or abs(hash(q))), "dataset": dataset, "question": q.strip(),
            "answers": list(answers), "supporting_titles": _dedup(p.get("title", "") for p in pos),
            "type": "single-hop", "level": "", "hops": 1, "paragraphs": paragraphs}


def conv_gold_passage(ex: dict, dataset: str) -> dict | None:
    """NQ-open com a passagem gold do Wikipedia-DPR (ex.: florin-hf/nq_open_gold)."""
    q = ex.get("question") or ex.get("query")
    answers = ex.get("answers") or ex.get("answer") or []
    if isinstance(answers, str):
        answers = [answers]
    text = ex.get("text") or ex.get("gold_document") or ex.get("context") or ""
    if not q or not answers or not text:
        return None
    return {"id": str(ex.get("example_id") or ex.get("id") or abs(hash(q))), "dataset": dataset, "question": q.strip(),
            "answers": list(answers), "supporting_titles": [ex.get("title", "")], "type": "single-hop", "level": "",
            "hops": 1, "paragraphs": [{"title": ex.get("title", ""), "text": text}]}


def _chunks(text: str, words: int = 100, max_chunks: int = 8) -> list[str]:
    w = (text or "").split()
    return [" ".join(w[i:i + words]) for i in range(0, min(len(w), words * max_chunks), words)]


def conv_trivia_rc(ex: dict, dataset: str = "triviaqa") -> dict | None:
    """TriviaQA rc.wikipedia: páginas de evidência cortadas em passagens de 100 palavras (como o wiki-DPR)."""
    ans = ex.get("answer") or {}
    answers = _dedup([ans.get("value", "")] + list(ans.get("aliases") or [])[:10])
    pages = ex.get("entity_pages") or {}
    titles, ctxs = pages.get("title") or [], pages.get("wiki_context") or []
    paragraphs = [{"title": t, "text": c} for t, ctx in zip(titles, ctxs) for c in _chunks(ctx)]
    if not ex.get("question") or not answers or not paragraphs:
        return None
    return {"id": str(ex.get("question_id") or abs(hash(ex["question"]))), "dataset": dataset,
            "question": ex["question"].strip(), "answers": answers, "supporting_titles": _dedup(titles),
            "type": "single-hop", "level": "", "hops": 1, "paragraphs": paragraphs}


CONVERTERS = {"gold_passage": conv_gold_passage, "trivia_rc": conv_trivia_rc}


def conv_flashrag_retrieved(ex: dict, dataset: str) -> dict:
    """Formato dos dados pré-processados do repositório oficial do RAG-Critic (top-k do E5 no wiki18)."""
    paragraphs = []
    for d in ex.get("retrieval_docs") or []:
        c = d.get("contents", "")
        title, _, text = c.partition("\n")
        paragraphs.append({"title": title.strip().strip('"'), "text": text.strip()})
    return {"id": str(ex.get("id") or abs(hash(ex["question"]))), "dataset": dataset, "question": ex["question"],
            "answers": list(ex.get("golden_answers") or []), "supporting_titles": [], "type": "", "level": "",
            "hops": None, "paragraphs": paragraphs, "given_docs": [p["title"] for p in paragraphs]}


def converter(dataset: str) -> Callable[[dict], dict | None]:
    if dataset in ("hotpotqa", "2wiki"):
        return lambda ex: conv_hotpot_like(ex, dataset)
    if dataset == "musique":
        return conv_musique
    if dataset in ("nq", "triviaqa"):
        return lambda ex: conv_tevatron(ex, dataset)
    raise ValueError(f"dataset desconhecido: {dataset}")


# ----------------------------------------------------------------------------- leitura das fontes
def _read_local(path: str) -> list[dict]:
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _stream_hf(src: dict, split: str, n: int, seed: int, conv) -> list[dict]:
    from datasets import load_dataset

    kw = {"split": split, "streaming": True}
    ds = load_dataset(src["repo"], src["config"], **kw) if src.get("config") else load_dataset(src["repo"], **kw)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out = []
    for ex in ds:
        r = conv(ex)
        if r is not None and r["answers"] and r["question"]:
            out.append(r)
        if len(out) >= n:
            break
    return out


def _direct_hf(src: dict, split: str, n: int, seed: int, conv) -> list[dict]:
    """Fallback: baixa os arquivos do repositório e lê à mão (parquet/json/jsonl, com ou sem gzip)."""
    from huggingface_hub import hf_hub_download, list_repo_files

    files = list_repo_files(src["repo"], repo_type="dataset")
    alias = {"validation": ["validation", "dev", "val"], "dev": ["dev", "validation", "val"], "train": ["train"],
             "test": ["test"]}[split]
    cands = [f for f in files if any(a in os.path.basename(f).lower() or f"/{a}" in f.lower() for a in alias)
             and f.lower().endswith((".parquet", ".json", ".jsonl", ".json.gz", ".jsonl.gz"))]
    if src.get("config"):
        pref = [f for f in cands if src["config"].lower() in f.lower()]
        cands = pref or cands
    if not cands:
        raise FileNotFoundError(f"nenhum arquivo de split {split} em {src['repo']}")
    rows: list[dict] = []
    for fn in sorted(cands):
        local = hf_hub_download(src["repo"], fn, repo_type="dataset")
        if fn.endswith(".parquet"):
            import pyarrow.parquet as pq

            rows.extend(pq.read_table(local).to_pylist())
        else:
            rows.extend(_read_local(local))
        if len(rows) >= 20 * n:
            break
    rng = random.Random(seed)
    rng.shuffle(rows)
    out = []
    for ex in rows:
        r = conv(ex)
        if r is not None and r["answers"] and r["question"]:
            out.append(r)
        if len(out) >= n:
            break
    return out


def fetch(dataset: str, which: str, n: int, seed: int, local: dict | None = None) -> tuple[list[dict], str]:
    """Busca ``n`` exemplos de ``which`` ('train' | 'eval') tentando as fontes em ordem."""
    log = get_logger()
    conv = converter(dataset)
    local = local or {}
    errors = []
    lp = local.get(which)
    if lp:
        if not os.path.exists(lp):
            errors.append(f"arquivo local não existe: {lp}")
        else:
            raw = _read_local(lp)
            random.Random(seed).shuffle(raw)
            fmt = local.get("format", "official")
            c = (lambda ex: conv_flashrag_retrieved(ex, dataset)) if fmt == "ragcritic" else conv
            out = [r for r in (c(ex) for ex in raw) if r is not None][:n]
            return out, f"local:{lp}"
    for src in HF_SOURCES.get(dataset, []):
        split = src[which]
        c = (lambda ex, f=CONVERTERS[src["conv"]]: f(ex, dataset)) if src.get("conv") else conv
        for mode, fn in (("stream", _stream_hf), ("direct", _direct_hf)):
            try:
                t0 = time.time()
                out = fn(src, split, n, seed, c)
                if out:
                    log.info(f"[{dataset}] {len(out)} exemplos de {src['repo']}:{split} ({mode}, {time.time() - t0:.0f}s)")
                    return out, f"hf:{src['repo']}:{split}:{mode}"
                errors.append(f"{src['repo']}:{split} ({mode}) veio vazio")
            except Exception as e:  # noqa: BLE001 — queremos tentar a próxima fonte
                errors.append(f"{src['repo']}:{split} ({mode}): {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError(f"[{dataset}] nenhuma fonte funcionou para '{which}':\n  - " + "\n  - ".join(errors))


# ----------------------------------------------------------------------------- corpus
def build_corpus(*groups: list[dict]) -> list[dict]:
    seen: dict = {}
    corpus: list[dict] = []
    for recs in groups:
        for r in recs:
            for p in r.get("paragraphs", []):
                text = (p.get("text") or "").strip()
                if not text:
                    continue
                key = (norm_title(p["title"]), text[:300])
                if key in seen:
                    continue
                seen[key] = len(corpus)
                corpus.append({"id": f"d{len(corpus)}", "title": p["title"], "text": text})
    return corpus


def _strip(recs: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if k != "paragraphs"} for r in recs]


def dataset_dir(cfg: dict, dataset: str) -> str:
    return os.path.join(cfg["paths"]["data"], dataset)


def is_prepared(cfg: dict, dataset: str) -> bool:
    d = dataset_dir(cfg, dataset)
    return all(os.path.exists(os.path.join(d, f)) for f in ("eval.jsonl", "corpus.jsonl", "meta.json"))


def prepare_dataset(cfg: dict, dataset: str, force: bool = False, need_train: bool = True) -> dict:
    log = get_logger()
    d = ensure_dir(dataset_dir(cfg, dataset))
    meta_path = os.path.join(d, "meta.json")
    if is_prepared(cfg, dataset) and not force:
        meta = read_json(meta_path, {})
        if meta.get("has_train") or not need_train:
            log.info(f"[{dataset}] já preparado ({meta.get('n_eval')} avaliação / {meta.get('n_train')} treino / "
                     f"{meta.get('n_docs')} parágrafos) — use --force para refazer")
            return meta
    sizes = cfg["data"]["scales"][cfg["data"]["scale"]]
    seed = int(cfg.get("seed", 13))
    local = (cfg["data"].get("local") or {}).get(dataset) or {}

    if dataset.startswith("toy_"):
        from .toy import make_toy

        eval_recs, train_recs, extra = make_toy(dataset, sizes["eval_pool"], sizes["train_pool"], sizes["extra_pool"], seed)
        src_eval = src_train = "toy"
    else:
        n_eval, n_extra = sizes["eval_pool"], sizes["extra_pool"]
        pool, src_eval = fetch(dataset, "eval", n_eval + n_extra, seed, local)
        eval_recs, extra = pool[:n_eval], pool[n_eval:]
        train_recs, src_train = [], "—"
        if need_train and sizes.get("train_pool", 0) > 0:
            try:
                train_recs, src_train = fetch(dataset, "train", sizes["train_pool"], seed, local)
            except RuntimeError as e:
                # sem split de treino: usa perguntas de avaliação DISJUNTAS das avaliadas
                log.warning(f"{e}\n[{dataset}] usando perguntas extras da avaliação (disjuntas) como treino")
                train_recs, extra = extra, []
                src_train = src_eval + " (disjunto)"
    corpus = build_corpus(eval_recs, train_recs, extra)
    write_jsonl(os.path.join(d, "eval.jsonl"), _strip(eval_recs))
    write_jsonl(os.path.join(d, "train.jsonl"), _strip(train_recs))
    write_jsonl(os.path.join(d, "corpus.jsonl"), corpus)
    meta = {"dataset": dataset, "source_eval": src_eval, "source_train": src_train, "n_eval": len(eval_recs),
            "n_train": len(train_recs), "n_extra": len(extra), "n_docs": len(corpus), "seed": seed,
            "scale": cfg["data"]["scale"], "has_train": bool(train_recs), "created": time.strftime("%Y-%m-%d %H:%M")}
    write_json(meta_path, meta)
    log.info(f"[{dataset}] avaliação={len(eval_recs)} treino={len(train_recs)} extra={len(extra)} "
             f"corpus={len(corpus)} parágrafos")
    return meta


# ----------------------------------------------------------------------------- acesso
_CORPORA: dict = {}


def load_corpus(cfg: dict, dataset: str) -> list[dict]:
    key = (cfg["paths"]["data"], dataset)
    if key not in _CORPORA:
        path = os.path.join(dataset_dir(cfg, dataset), "corpus.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} não existe — rode `python -m ragbench prepare` antes")
        _CORPORA[key] = read_jsonl(path)
    return _CORPORA[key]


def load_records(cfg: dict, dataset: str, split: str, n: int | None = None) -> list[dict]:
    path = os.path.join(dataset_dir(cfg, dataset), f"{split}.jsonl")
    recs = read_jsonl(path)
    if not recs and split == "eval":
        raise FileNotFoundError(f"{path} não existe ou está vazio — rode `python -m ragbench prepare`")
    return recs[:n] if n else recs


def clear_cache() -> None:
    _CORPORA.clear()


__all__ = ["prepare_dataset", "load_corpus", "load_records", "build_corpus", "fetch", "DATASET_NAMES",
           "dataset_dir", "is_prepared", "conv_hotpot_like", "conv_musique", "conv_tevatron",
           "conv_flashrag_retrieved"]
