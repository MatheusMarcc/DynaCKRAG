"""Utilidades gerais: IO jsonl, normalização de títulos e tokens de conteúdo."""
from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Iterable

_WS = re.compile(r"\s+")
_WORD = re.compile(r"\w+", re.UNICODE)

STOPWORDS = set(
    """a an the of in on at to for from by with and or is was are were be been being
    who what when where which why how whose whom did does do done that this these those
    as it its his her their he she they them there than then also into about over under
    both either same more most less least first last one two many much""".split()
)


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_title(t: str) -> str:
    """Título normalizado: usado para deduplicar docs e calcular o recall de suporte."""
    t = unicodedata.normalize("NFKC", t or "").lower().strip()
    t = t.strip('"').strip()
    return _WS.sub(" ", t)


def content_tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall((text or "").lower()) if w not in STOPWORDS]


def passage_str(doc: dict) -> str:
    """Formato "Título"\\ntexto — o mesmo do corpus wiki do FlashRAG / RAG-Critic."""
    return f"\"{doc['title']}\"\n{doc['text']}"


def numbered_passages(docs: list[dict], max_chars: int = 1200) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        text = d["text"] if len(d["text"]) <= max_chars else d["text"][:max_chars] + "…"
        out.append(f"[{i}] {d['title']}: {text}")
    return "\n".join(out)
