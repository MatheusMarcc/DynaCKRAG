"""Tabelas no padrão de apresentação (o da Tabela 1 do DynaKRAG), UMA POR ARTIGO, cada uma só com as métricas
que o próprio artigo usa — nada de aplicar a métrica de um artigo ao método de outro:

* DynaKRAG   — EM↑, F1↑ e Tok.↓ por dataset (Tabela 1 dele): Fixed-K, os baselines que já rodaram e o DynaKRAG.
* RAG-Critic — EM↑ e F1↑ por dataset e Overall F1 com o ganho sobre o Standard RAG (Tabela 3 dele).

Saídas em ``ragbench_results/<perfil>/``: ``TABELAS.md``, ``tabelas.tex`` (booktabs) e ``figs/tabela_*.png``
(prontas para slide). Negrito = melhor da coluna; sublinhado = segundo melhor (no .md, itálico).
"""
from __future__ import annotations

import math
import os

from .analysis import best_fixed_k, load_eval, result_files
from .core.datasets import DATASET_NAMES
from .core.io import read_jsonl
from .core.text import best_em_f1
from .dynakrag.state import control_signature, row_control

FIXED = "@fixed_k"                    # resolvido por dataset: o melhor K da grade, como no artigo
DS_ORDER = ("nq", "triviaqa", "hotpotqa", "2wiki", "musique")
# linhas de cada tabela, na ordem do artigo: (nome do arquivo de resultado, rótulo, é o método proposto?)
DK_ROWS = [(FIXED, "Fixed-K", False), ("ircot", "IRCoT", False), ("s2g_rag", "S2G-RAG", False), ("corag_s3", "CoRAG-s3", False),
           ("adaptive_rag", "Adaptive-RAG", False), ("par2_rag", "PAR²-RAG", False), ("crag", "CRAG", False),
           ("self_ask", "Self-Ask+Search", False), ("dynakrag", "DynaKRAG", True)]
RC_ROWS = [("standard_rag", "Standard RAG", False), ("self_refine", "Self-Refine", False), ("flare", "FLARE", False),
           ("metarag", "MetaRAG", False), ("ragcritic", "RAG-Critic", True)]


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else None


def _tok(r: dict) -> float:
    from .compare import _tokens

    return _tokens(r)


def load(cfg: dict) -> dict:
    """{dataset: {"artigo/método": {id: linha}}} com EM e F1 garantidos em cada linha."""
    out = {}
    for ds, files in result_files(cfg).items():
        recs = load_eval(cfg, ds)
        rows_by = {}
        for key, path in files.items():
            rows = {r["id"]: r for r in read_jsonl(path) if not r.get("error") and "id" in r}
            for i, r in rows.items():
                golds = r.get("answers") or (recs.get(i) or {}).get("answers") or []
                if r.get("em") is None or r.get("f1") is None:
                    r["em"], r["f1"] = best_em_f1(str(r.get("pred") or ""), golds)
            if rows:
                rows_by[key] = rows
        if rows_by:
            out[ds] = rows_by
    return out


def _datasets(data: dict, prefix: str) -> list[str]:
    have = [d for d in data if any(k.startswith(prefix) for k in data[d])]
    return [d for d in DS_ORDER if d in have] + sorted(d for d in have if d not in DS_ORDER)


def _stats(rows: dict | None) -> dict:
    if not rows:
        return {}
    rs = list(rows.values())
    return {"n": len(rs), "em": _mean([float(r["em"]) for r in rs]), "f1": _mean([float(r["f1"]) for r in rs]),
            "tok": _mean([_tok(r) for r in rs])}


# ----------------------------------------------------------------------------- destaque (melhor / segundo)
def _rank(values: list, higher: bool) -> list[str]:
    """'b' para o melhor (empates incluídos), 'u' para o segundo melhor valor distinto, '' para o resto."""
    vals = sorted({round(v, 6) for v in values if v is not None}, reverse=higher)
    tie = sum(1 for v in values if v is not None and round(v, 6) == vals[0]) > 1 if vals else False
    out = []
    for v in values:
        if v is None or not vals:
            out.append("")
        elif round(v, 6) == vals[0]:
            out.append("b")
        elif len(vals) > 2 and not tie and round(v, 6) == vals[1]:   # sem "segundo" com empate no topo ou só 2 valores
            out.append("u")
        else:
            out.append("")
    return out


def _fmt(v, kind: str) -> str:
    if v is None:
        return "—"
    if kind == "frac":
        return f"{v:.4f}"
    if kind == "tok":
        return f"{v:.1f}"
    return f"{v:.2f}"


def build_tables(cfg: dict) -> list[dict]:
    """Uma tabela por artigo, em formato neutro: cabeçalho em dois níveis + linhas de células (texto, destaque)."""
    data = load(cfg)
    model = str(cfg["models"]["generator"])
    tabs = []

    # ---- DynaKRAG (Tabela 1 do artigo): EM, F1, Tok. por dataset
    dss = _datasets(data, "dynakrag/")
    fixed = {d: best_fixed_k({k: [{"f": float(r["f1"])} for r in v.values()] for k, v in data[d].items()}) for d in dss}
    st = {(name, d): _stats(data[d].get(fixed[d] if name == FIXED else f"dynakrag/{name}") if (name != FIXED or fixed[d]) else None)
          for name, _, _ in DK_ROWS for d in dss}
    rows = [r for r in DK_ROWS if any(st[(r[0], d)] for d in dss)]
    if rows:
        names = [DATASET_NAMES.get(d, d) for d in dss]
        mk = {(d, f): _rank([st[(r[0], d)].get(f) for r in rows], f != "tok") for d in dss for f in ("em", "f1", "tok")}
        # linha do DynaKRAG gerada com outras regras de parada/sonda: fica marcada com † na tabela e no rodapé
        sig = control_signature(cfg)
        old_ctrl = {d: sorted({row_control(r) for r in (data[d].get("dynakrag/dynakrag") or {}).values()} - {sig}) for d in dss}
        stale = [d for d in dss if old_ctrl[d]]
        t = {"id": "dynakrag", "paper": "DynaKRAG", "title": "DynaKRAG — acerto e custo (formato da Tabela 1 do artigo)",
             "groups": [("Model", 1), ("Method", 1)] + [(n, 3) for n in names], "sub": ["", ""] + ["EM↑", "F1↑", "Tok.↓"] * len(dss),
             "rows": [], "hl": [], "n": {d: max((st[(r[0], d)].get("n", 0) for r in rows), default=0) for d in dss},
             "note": "EM e F1 medem o acerto; Tok. é o total de tokens por pergunta, somando todas as chamadas ao modelo. "
                     "*Fixed-K* é o melhor K da grade em cada dataset ("
                     + ", ".join(f"{DATASET_NAMES.get(d, d)}: K={fixed[d].split('fixed_k')[-1]}" for d in dss if fixed[d])
                     + "), como no artigo."}
        if stale:
            t["warn"] = ("† A linha DynaKRAG de " + ", ".join(DATASET_NAMES.get(d, d) for d in stale) + " foi gerada com as regras "
                         "de controle antigas (" + "; ".join(sorted({c for d in stale for c in old_ctrl[d]})) + "), que não "
                         "reproduzem a trajetória do artigo (Tabelas 6–8 dele): não use esses números como resultado do DynaKRAG. "
                         "Rode o DynaKRAG de novo — os resultados antigos ficam guardados em `ragbench_results/logs/controle_anterior/`.")
        for j, (name, label, main) in enumerate(rows):
            cells = [(model if j == 0 else "", ""), (label + (" †" if name == "dynakrag" and stale else ""), "b" if main else "")]
            for d in dss:
                for f, kind in (("em", "frac"), ("f1", "frac"), ("tok", "tok")):
                    cells.append((_fmt(st[(name, d)].get(f), kind), mk[(d, f)][j]))
            t["rows"].append(cells)
            t["hl"].append(main)
        tabs.append(t)

    # ---- RAG-Critic (Tabela 3 do artigo): EM e F1 por dataset + Overall F1 com o ganho sobre o Standard RAG
    dss = _datasets(data, "ragcritic/")
    st = {(name, d): _stats(data[d].get(f"ragcritic/{name}")) for name, _, _ in RC_ROWS for d in dss}
    rows = [r for r in RC_ROWS if any(st[(r[0], d)] for d in dss)]
    if rows:
        names = [DATASET_NAMES.get(d, d) for d in dss]
        full = [d for d in dss if all(st[(r[0], d)] for r in rows)] or dss      # Overall só onde todos rodaram
        ov = {r[0]: _mean([st[(r[0], d)].get("f1") for d in full]) for r in rows}
        mk = {(d, f): _rank([st[(r[0], d)].get(f) for r in rows], True) for d in dss for f in ("em", "f1")}
        mov = _rank([ov[r[0]] for r in rows], True)
        t = {"id": "ragcritic", "paper": "RAG-Critic", "title": "RAG-Critic — acerto por dataset (formato da Tabela 3 do artigo)",
             "groups": [("Model", 1), ("Method", 1)] + [(n, 2) for n in names] + [("Overall", 1)],
             "sub": ["", ""] + ["EM↑", "F1↑"] * len(dss) + ["F1↑"], "rows": [], "hl": [],
             "n": {d: max((st[(r[0], d)].get("n", 0) for r in rows), default=0) for d in dss},
             "note": "*Overall* é a média do F1 nos datasets (" + ", ".join(DATASET_NAMES.get(d, d) for d in full)
                     + "); entre parênteses, o ganho sobre o Standard RAG em pontos percentuais, como no artigo."}
        for j, (name, label, main) in enumerate(rows):
            cells = [(model if j == 0 else "", ""), (label, "b" if main else "")]
            for d in dss:
                for f in ("em", "f1"):
                    cells.append((_fmt(st[(name, d)].get(f), "frac"), mk[(d, f)][j]))
            gain = None if (name == "standard_rag" or ov[name] is None or ov.get("standard_rag") is None) else ov[name] - ov["standard_rag"]
            cells.append((_fmt(ov[name], "frac") + (f" ({100 * gain:+.1f})" if gain is not None else ""), mov[j]))
            t["rows"].append(cells)
            t["hl"].append(main)
        tabs.append(t)

    return tabs


# ----------------------------------------------------------------------------- saídas
def _md(t: dict) -> str:
    head = []
    for name, span in t["groups"]:
        head += [name] + [""] * (span - 1)
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * 2 + ["---:"] * (len(head) - 2)) + "|",
             "| " + " | ".join(t["sub"]) + " |"]
    for cells in t["rows"]:
        lines.append("| " + " | ".join(f"**{x}**" if m == "b" else (f"_{x}_" if m == "u" else x) for x, m in cells) + " |")
    return "\n".join(lines)


def _tex_escape(s: str) -> str:
    rep = {"↑": r"$\uparrow$", "↓": r"$\downarrow$", "Δ": r"$\Delta$", "†": r"$^\dagger$", "—": "--", "_": r"\_", "%": r"\%",
           "−": "-"}
    for a, b in rep.items():
        s = s.replace(a, b)
    return s


def _tex(t: dict) -> str:
    ncol = sum(s for _, s in t["groups"])
    out = [r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{ll" + "r" * (ncol - 2) + "}", r"\toprule"]
    top, rules, pos = [], [], 1
    for name, span in t["groups"]:
        top.append(rf"\multicolumn{{{span}}}{{c}}{{\textbf{{{_tex_escape(name)}}}}}" if span > 1 else rf"\textbf{{{_tex_escape(name)}}}")
        if span > 1:
            rules.append(rf"\cmidrule(lr){{{pos}-{pos + span - 1}}}")
        pos += span
    out += [" & ".join(top) + r" \\", " ".join(rules), " & ".join(rf"\textbf{{{_tex_escape(s)}}}" if s else "" for s in t["sub"]) + r" \\",
            r"\midrule"]
    for cells, hl in zip(t["rows"], t["hl"]):
        row = " & ".join(rf"\textbf{{{_tex_escape(x)}}}" if m == "b" else (rf"\underline{{{_tex_escape(x)}}}" if m == "u" else _tex_escape(x))
                         for x, m in cells)
        if cells[0][0] and len(out) and not out[-1].startswith(r"\midrule"):
            out.append(r"\midrule")
        out.append((r"\rowcolor{red!10} " if hl else "") + row + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{_tex_escape(t['title'])}}}", r"\end{table}", ""]
    return "\n".join(out)


def _png(t: dict, path: str) -> str | None:
    """Desenha a tabela no estilo dos artigos (réguas horizontais, negrito, sublinhado, linha do método destacada)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 — sem matplotlib, ficam o .md e o .tex
        return None
    ncol = sum(s for _, s in t["groups"])
    widths = [max([len(t["sub"][j])] + [len(r[j][0]) for r in t["rows"]]) for j in range(ncol)]
    pos = 0
    for name, span in t["groups"]:                    # o rótulo do grupo também precisa caber
        if span == 1:
            widths[pos] = max(widths[pos], len(name))
        pos += span
    widths = [w * 0.105 + 0.28 for w in widths]
    xs = [sum(widths[:j]) for j in range(ncol + 1)]
    rh, nrow = 0.30, len(t["rows"])
    W, H = xs[-1], rh * (nrow + 2) + 0.16
    fig = plt.figure(figsize=(W, H), dpi=220)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    font = {"family": "DejaVu Serif", "size": 10}
    y_top, y_grp, y_sub = H - 0.06, H - 0.06 - rh / 2, H - 0.06 - rh * 1.5

    def cx(j, span=1, left=False):
        return xs[j] + 0.10 if left else (xs[j] + xs[j + span]) / 2

    ax.plot([0, W], [y_top, y_top], color="black", lw=1.1)
    pos, under = 0, []
    for name, span in t["groups"]:
        if name:
            ax.text(cx(pos, span, left=(pos < 2 and span == 1)), y_grp if span > 1 else (y_grp + y_sub) / 2, name,
                    ha="left" if (pos < 2 and span == 1) else "center", va="center", fontweight="bold", **font)
            if span > 1:
                ax.plot([xs[pos] + 0.08, xs[pos + span] - 0.08], [y_grp - rh / 2] * 2, color="black", lw=0.6)
        pos += span
    for j, s in enumerate(t["sub"]):
        if s:
            ax.text(cx(j), y_sub, s, ha="center", va="center", fontweight="bold", **font)
    y0 = y_sub - rh / 2
    ax.plot([0, W], [y0, y0], color="black", lw=0.7)
    for i, (cells, hl) in enumerate(zip(t["rows"], t["hl"])):
        y = y0 - rh * (i + 0.5)
        if i and cells[0][0]:                           # novo bloco (outro modelo): régua fina, como nos artigos
            ax.plot([0, W], [y + rh / 2] * 2, color="black", lw=0.5, zorder=3)
        if hl:
            ax.add_patch(plt.Rectangle((xs[1], y - rh / 2), W - xs[1], rh, color="#fbdcdf", lw=0, zorder=0))
        for j, (x, m) in enumerate(cells):
            left = j < 2
            tx = ax.text(cx(j, left=left), y, x, ha="left" if left else "center", va="center",
                         fontweight="bold" if m == "b" else "normal", **font)
            if m == "u":
                under.append(tx)
    yb = y0 - rh * nrow
    ax.plot([0, W], [yb, yb], color="black", lw=1.1)
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for tx in under:                                   # sublinhado do segundo melhor
        bb = tx.get_window_extent(fig.canvas.get_renderer())
        (x0, y0_), (x1, _) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y0))
        ax.plot([x0, x1], [y0_ - 0.012, y0_ - 0.012], color="black", lw=0.6)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def write_tables(cfg: dict) -> str | None:
    """Gera ``TABELAS.md``, ``tabelas.tex`` e as figuras. Devolve o caminho do .md (ou None, sem resultados)."""
    tabs = build_tables(cfg)
    res = cfg["paths"]["results"]
    if not tabs:
        return None
    ns = sorted({n for t in tabs for n in t["n"].values() if n})
    md = [f"# Tabelas no padrão dos artigos ({cfg['profile']})", "",
          "Uma tabela por artigo, cada uma só com as métricas que o próprio artigo usa. "
          f"Gerador: `{cfg['models']['generator']}`. Perguntas por dataset: {', '.join(str(n) for n in ns)}. "
          "EM e F1 são frações (0.4684 = 46,84%).", "",
          "**Negrito**: melhor da coluna. _Itálico_ (sublinhado nas figuras e no LaTeX): segundo melhor. ↑ maior é "
          "melhor; ↓ menor é melhor. A linha do método proposto pelo artigo vem destacada. As figuras em `figs/` e o "
          "`tabelas.tex` têm as mesmas tabelas, prontas para slide.", ""]
    tex = ["% Gerado por ragbench (python -m ragbench report). Requer: \\usepackage{booktabs} \\usepackage[table]{xcolor}", ""]
    for t in tabs:
        png = _png(t, os.path.join(res, "figs", f"tabela_{t['id']}.png"))
        md += [f"## {t['title']}", ""]
        if png:
            md += [f"![{t['title']}](figs/tabela_{t['id']}.png)", ""]
        md += [_md(t), "", t.get("note", ""), ""]
        if t.get("warn"):
            md += ["> **Atenção.** " + t["warn"], ""]
        tex.append(_tex(t))
    md += ["## Onde estão as outras métricas de cada artigo", "",
           "* **RAG-Critic** — Identificação e Tag1/Tag2 F1 do crítico no RAG-Error Bench (Tabela 5) e as ablações: "
           "`ragcritic/REPORT.md`.",
           "* **DynaKRAG** — ablações (Tabela 2), limite de buscas (Tabela 6, Fig. 3), ações do controlador (Tabela 7, "
           "Fig. 4), sonda de suficiência (Tabela 8) e por tipo de pergunta (Fig. 5): `dynakrag/REPORT.md`.",
           "* **Onde cada método erra**, pergunta por pergunta: `FALHAS.md` e as abas 2 e 3 do painel.", ""]
    os.makedirs(res, exist_ok=True)
    path = os.path.join(res, "TABELAS.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(res, "tabelas.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    return path
