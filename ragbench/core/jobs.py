"""Agendador de jobs dos benchmarks, em ordem de prioridade (tier 1 → 2 → 3).

Cada artigo declara os seus jobs (``dynakrag.bench.jobs`` e ``ragcritic.bench.jobs``). O agendador
roda tier por tier, alternando os artigos, para que os resultados mais importantes saiam primeiro
e o relatório possa ser gerado a qualquer momento com o que já terminou. Um job que falha é
registrado em ``status.json`` e o agendador segue para o próximo.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass
from typing import Callable

from .io import get_logger, read_json, write_json


@dataclass
class Job:
    paper: str
    tier: int
    dataset: str
    name: str
    run: Callable[[], object]
    desc: str = ""

    @property
    def key(self) -> str:
        return f"{self.paper}/{self.dataset}/{self.name}"


def run_jobs(jobs: list[Job], status_path: str, tiers=(1, 2, 3), stop_on_llm_error: bool = True) -> dict:
    from .llm import LLMError

    log = get_logger()
    status = read_json(status_path, {}) or {}
    order = sorted(enumerate(jobs), key=lambda t: (t[1].tier, t[1].paper != "ragcritic", t[0]))
    todo = [j for _, j in order if j.tier in tiers]
    log.info(f"{len(todo)} jobs na fila (tiers {list(tiers)})")
    for i, job in enumerate(todo, 1):
        log.info(f"▶ [{i}/{len(todo)}] tier {job.tier} · {job.key}  {job.desc}")
        t0 = time.time()
        status[job.key] = {"state": "running", "tier": job.tier, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        write_json(status_path, status)
        try:
            job.run()
            status[job.key] = {"state": "done", "tier": job.tier, "secs": round(time.time() - t0, 1),
                               "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            log.info(f"✔ {job.key} em {(time.time() - t0) / 60:.1f} min")
        except LLMError as e:
            status[job.key] = {"state": "failed", "tier": job.tier, "error": str(e)}
            write_json(status_path, status)
            log.error(f"✖ {job.key}: {e}")
            if stop_on_llm_error:
                log.error("O LLM ficou indisponível — interrompendo. Rode de novo o mesmo comando para continuar.")
                raise
        except Exception as e:  # noqa: BLE001
            status[job.key] = {"state": "failed", "tier": job.tier, "error": f"{type(e).__name__}: {e}",
                               "trace": traceback.format_exc(limit=6)}
            log.error(f"✖ {job.key}: {type(e).__name__}: {e} (seguindo para o próximo job)")
        write_json(status_path, status)
    return status
