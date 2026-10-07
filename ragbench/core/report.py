"""Ajudantes de relatório: tabelas markdown e figuras (matplotlib, backend sem janela)."""
from __future__ import annotations

import math
import os

from .io import ensure_dir


def fmt(x, nd: int = 4, pct: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if isinstance(x, str):
        return x
    if pct:
        return f"{100 * x:.1f}"
    if isinstance(x, int) or (isinstance(x, float) and x.is_integer() and abs(x) >= 100):
        return f"{int(round(x))}"
    if abs(x) >= 100:
        return f"{x:.1f}"
    return f"{x:.{nd}f}"


def md_table(headers: list[str], rows: list[list], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(headers) - 1))
    sep = ["---:" if a == "r" else (":---:" if a == "c" else "---") for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def bold_best(values: list[float], higher: bool = True) -> list[bool]:
    vals = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return [False] * len(values)
    best = max(vals) if higher else min(vals)
    return [v is not None and not (isinstance(v, float) and math.isnan(v)) and abs(v - best) < 1e-12 for v in values]


def plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt

    _plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    return _plt


def savefig(fig, path: str) -> str:
    ensure_dir(os.path.dirname(path) or ".")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    import matplotlib.pyplot as _plt

    _plt.close(fig)
    return path


PALETTE = ["#2a6f97", "#e76f51", "#7b9e3c", "#9c6ade", "#f4a261", "#2a9d8f", "#8d99ae", "#d62828", "#6c757d",
           "#264653"]
