"""Configuração: ``configs/base.yaml`` + ``configs/profiles/<perfil>.yaml`` + overrides ``--set a.b=valor``."""
from __future__ import annotations

import copy
import os

import yaml

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_DIR = os.path.join(PKG_DIR, "configs")
PROFILES = ["smoke", "mini", "quick", "standard"]


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_path(cfg: dict, dotted: str, value) -> None:
    cur = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def get_path(cfg: dict, dotted: str, default=None):
    cur = cfg
    for p in dotted.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def load_config(profile: str = "quick", overrides: list[str] | None = None, root: str | None = None) -> dict:
    with open(os.path.join(CONFIG_DIR, "base.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prof_path = os.path.join(CONFIG_DIR, "profiles", f"{profile}.yaml")
    if not os.path.exists(prof_path):
        raise SystemExit(f"perfil desconhecido: {profile} (opções: {', '.join(PROFILES)})")
    with open(prof_path, encoding="utf-8") as f:
        cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        if "=" not in ov:
            raise SystemExit(f"--set espera chave=valor, recebi: {ov}")
        k, v = ov.split("=", 1)
        set_path(cfg, k.strip(), yaml.safe_load(v))
    cfg["profile"] = profile
    root = os.path.abspath(root or os.getcwd())
    cfg["root"] = root
    p = cfg["paths"]
    base_data = p["data"] if os.path.isabs(p["data"]) else os.path.join(root, p["data"])
    base_res = p["results"] if os.path.isabs(p["results"]) else os.path.join(root, p["results"])
    cache = p["cache"] if (p["cache"] == ":memory:" or os.path.isabs(p["cache"])) else os.path.join(root, p["cache"])
    p["data_root"] = base_data
    p["data"] = os.path.join(base_data, cfg["data"]["scale"])
    p["results"] = os.path.join(base_res, profile)
    p["cache"] = cache
    p["logs"] = os.path.join(base_res, "logs")
    return cfg


def retrieval_spec(cfg: dict, paper: str) -> dict:
    r = cfg["retrieval"]
    spec = dict(r[paper])
    for k in ("device", "batch_size", "max_seq_length", "embed_backend"):
        spec.setdefault(k, r.get(k))
    spec.setdefault("ollama_url", cfg.get("llm", {}).get("base_url", "http://localhost:11434"))
    if cfg.get("llm", {}).get("backend") != "ollama" and spec.get("embed_backend") in (None, "auto"):
        spec["embed_backend"] = "st"      # sem Ollama (mock, API OpenAI): nada de /api/embed
    return spec


PAPERS = ("ragcritic", "dynakrag")


def paper_datasets(cfg: dict, paper: str) -> list[str]:
    """Datasets em que o artigo é avaliado (os do próprio artigo), sem repetir."""
    return list(dict.fromkeys(cfg.get(paper, {}).get("datasets") or []))


def shared_datasets(cfg: dict) -> list[str]:
    """Datasets usados por todos os artigos: só neles os métodos respondem às mesmas perguntas."""
    first, *rest = [paper_datasets(cfg, p) for p in PAPERS]
    return [ds for ds in first if all(ds in other for other in rest)]
