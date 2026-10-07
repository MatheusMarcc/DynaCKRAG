"""Médias das métricas por método."""
from __future__ import annotations

import math
from statistics import mean


def safe_mean(xs) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return mean(xs) if xs else float("nan")


def summarize(rows: list[dict], keys=("em", "f1", "tokens", "llm_calls", "n_ret", "steps", "secs", "llm_seconds")) -> dict:
    out = {"n": len(rows)}
    for k in keys:
        vals = [r.get(k) for r in rows if isinstance(r.get(k), (int, float))]
        out[k] = safe_mean(vals) if vals else float("nan")
    if rows:
        out["n_errors"] = sum(bool(r.get("error")) for r in rows)
    return out
