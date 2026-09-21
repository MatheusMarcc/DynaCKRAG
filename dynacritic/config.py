"""Carrega o config.yaml e permite sobrescrever chaves pela linha de comando (--set a.b=valor)."""
from __future__ import annotations

import copy
import os

import yaml


def load_config(path: str = "config.yaml", overrides: list[str] | None = None) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        val = yaml.safe_load(raw)
        if isinstance(val, (dict, list)) and not raw.strip().startswith(("{", "[")):
            val = raw  # ex.: ":memory:" seria lido como dicionário pelo YAML
        node[parts[-1]] = val
    cfg = copy.deepcopy(cfg)
    ddir = cfg["data"]["dir"]
    os.makedirs(ddir, exist_ok=True)
    cfg["paths"] = {
        "train": os.path.join(ddir, "train.jsonl"),
        "dev": os.path.join(ddir, "dev.jsonl"),
        "corpus": os.path.join(ddir, "corpus.jsonl"),
        "index": os.path.join(ddir, "index"),
        "labels": os.path.join(ddir, "labels.npz"),
        "runs": os.path.join(ddir, "runs"),
        "rf": cfg["loop"].get("rf_path") or os.path.join(ddir, "controller_rf.pkl"),
    }
    if not cfg["llm"].get("cache_path"):
        cfg["llm"]["cache_path"] = os.path.join(ddir, "llm_cache.sqlite")
    return cfg
