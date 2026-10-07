"""Suíte de benchmark do RAG-Critic (isolado): Tabela 3 (principal), Tabela 4 (ablações viáveis),
Tabela 6 (iterações N), Tabelas 5/7 + auditoria de formato (RAG-Error Bench) e Fig. 6 (distribuição de erros).

Saídas: ``<results>/ragcritic/<dataset>/<método>.jsonl`` e ``<results>/ragcritic/errorbench/``.
"""
from __future__ import annotations

from functools import partial

from ..config import paper_datasets
from ..core.datasets import is_prepared
from ..core.io import get_logger
from ..core.jobs import Job
from . import baselines as BL
from .errorbench import format_artifact_stats, run_errorbench
from .workflow import Ctx, run_standard, run_variant, variants

PAPER = "ragcritic"


def run_method(cfg: dict, dataset: str, method: str, llm, limit: int | None = None):
    ctx = Ctx(cfg, dataset, llm, limit)
    if method == "standard_rag":
        return run_standard(ctx)
    if method in BL.BASELINES:
        return BL.run_baseline(ctx, method)
    vs = variants(cfg)
    if method not in vs:
        raise ValueError(f"método desconhecido: {method} (opções: standard_rag, {', '.join(BL.BASELINES)}, {', '.join(vs)})")
    return run_variant(ctx, vs[method])


def method_names(cfg: dict) -> list[tuple[str, int]]:
    rc = cfg["ragcritic"]
    out = [("standard_rag", 1), ("ragcritic", 1)]
    vs = variants(cfg)
    abl = rc.get("ablations", [])
    if int(rc.get("iterations", 1)) >= 2:
        out.append(("ragcritic_n2", 2))
    for name in vs:
        if name in ("ragcritic", "ragcritic_n2"):
            continue
        if name.replace("ragcritic_", "") in abl:      # ablações da Tabela 4 do artigo
            out.append((name, 2))
    for m in rc.get("methods", []):
        if m in BL.BASELINES:
            out.append((m, 3))
    return out


def jobs(cfg: dict, llm, datasets: list[str] | None = None, methods: list[str] | None = None) -> list[Job]:
    log = get_logger()
    rc = cfg["ragcritic"]
    out: list[Job] = []
    if not methods or "errorbench" in methods:
        out.append(Job(PAPER, 1, "errorbench", "rag-critic-3b",
                       partial(lambda: (format_artifact_stats(cfg), run_errorbench(cfg, llm, "finetuned"))),
                       "RAG-Critic-3B no RAG-Error Bench + auditoria de formato"))
        for m in rc["errorbench"].get("zeroshot_models", []):
            out.append(Job(PAPER, 2, "errorbench", f"zeroshot-{m}", partial(run_errorbench, cfg, llm, "zeroshot", m),
                           "LLM zero-shot com o Prompt E"))
    for ds in datasets or paper_datasets(cfg, PAPER):
        if not is_prepared(cfg, ds):
            log.warning(f"[RAG-Critic] {ds} não está preparado — pulando (rode prepare)")
            continue
        for name, tier in method_names(cfg):
            if methods and name not in methods:
                continue
            out.append(Job(PAPER, tier, ds, name, partial(run_method, cfg, ds, name, llm)))
    return out
