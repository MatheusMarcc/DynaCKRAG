"""Tabela comparativa de todas as execuções em data/runs/."""
from __future__ import annotations

import glob
import os
from collections import Counter

from .utils import read_jsonl


def _summ(rows: list[dict]) -> dict:
    n = max(1, len(rows))
    avg = lambda k: sum(r.get(k, 0) or 0 for r in rows) / n  # noqa: E731
    u = [r.get("usage", {}) for r in rows]
    return {
        "n": len(rows), "EM": avg("em"), "F1": avg("f1"), "SR": avg("sr"),
        "tokens": sum(x.get("prompt_tokens", 0) + x.get("completion_tokens", 0) for x in u) / n,
        "llm_calls": sum(x.get("calls", 0) for x in u) / n,
        "ret_calls": avg("n_ret") if rows and "n_ret" in rows[0] else float("nan"),
        "steps": avg("steps") if rows and "steps" in rows[0] else float("nan"),
    }


def evaluate_runs(cfg: dict) -> None:
    files = sorted(glob.glob(os.path.join(cfg["paths"]["runs"], "*", "*.jsonl")))
    files = [f for f in files if not f.endswith("critic.jsonl")]
    if not files:
        print("nenhuma execução encontrada em", cfg["paths"]["runs"])
        return
    head = f"{'execução':38s} {'n':>5s} {'EM':>7s} {'F1':>7s} {'SR':>6s} {'tokens':>8s} {'LLM':>5s} {'ret':>5s} {'passos':>6s}"
    print(head)
    print("-" * len(head))
    for f in files:
        rows = read_jsonl(f)
        s = _summ(rows)
        name = os.path.relpath(f, cfg["paths"]["runs"]).replace(os.sep, "/").replace("/preds.jsonl", "")
        print(f"{name:38s} {s['n']:5d} {s['EM']:7.4f} {s['F1']:7.4f} {s['SR']:6.3f} {s['tokens']:8.0f} "
              f"{s['llm_calls']:5.2f} {s['ret_calls']:5.2f} {s['steps']:6.2f}")
    # composição de ações (equivalente à Figura 4 do DynaKRAG)
    for f in files:
        if not f.endswith("preds.jsonl"):
            continue
        rows = read_jsonl(f)
        c = Counter(a for r in rows for a in r.get("history", []))
        if c:
            n = len(rows)
            comp = ", ".join(f"{a}={v / n:.2f}" for a, v in sorted(c.items(), key=lambda t: -t[1]))
            print(f"  ações/exemplo  {os.path.basename(os.path.dirname(f))}: {comp}")
