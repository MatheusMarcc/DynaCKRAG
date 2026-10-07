"""Relatório geral: um índice com o estado dos jobs e links para os relatórios de cada artigo."""
from __future__ import annotations

import os
import time

from .config import PAPERS
from .core.io import read_json
from .core.tokens import tokenizer_kind


def build_all(cfg: dict) -> dict:
    from .dynakrag.report import build as build_dk
    from .ragcritic.report import build as build_rc

    from .analysis import write_reports as build_fail
    from .compare import build as build_cmp
    from .tables import write_tables as build_tables

    paths = {}
    for name, fn in (("dynakrag", build_dk), ("ragcritic", build_rc), ("comparacao", build_cmp),
                     ("falhas", lambda c: build_fail(c)["md"]), ("tabelas", build_tables)):
        try:
            paths[name] = fn(cfg)
        except Exception as e:  # noqa: BLE001 — um relatório quebrado não impede o outro
            import traceback

            paths[name] = f"ERRO: {type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
    res = cfg["paths"]["results"]
    status = read_json(os.path.join(res, "status.json"), {}) or {}
    status = {k: v for k, v in status.items() if k.split("/", 1)[0] in PAPERS}      # só os jobs dos artigos que existem hoje
    rows = []
    for key, st in sorted(status.items(), key=lambda kv: (kv[1].get("tier", 9), kv[0])):
        mins = f"{st.get('secs', 0) / 60:.1f}" if st.get("secs") else "—"
        rows.append(f"| {st.get('tier', '?')} | `{key}` | {st.get('state')} | {mins} | {(st.get('error') or '')[:120]} |")
    lines = [f"# ragbench — resultados ({cfg['profile']})", "",
             f"Gerado em {time.strftime('%Y-%m-%d %H:%M')}. Contagem de tokens: {tokenizer_kind()}.", "",
             "* [**Tabelas no padrão dos artigos**](TABELAS.md): uma tabela por artigo, cada uma só com as métricas do "
             "próprio artigo. Figuras prontas para slide em `figs/` e LaTeX em `tabelas.tex`.",
             "* [**Comparação com os artigos**](COMPARACAO.md): cada afirmação dos artigos, com o Δ da reprodução ao "
             "lado do Δ do artigo e o veredito (mesmo sinal / sinal oposto)",
             "* [**Análise de falhas**](FALHAS.md): em quais perguntas cada método falha, em que etapa, e quais "
             "falhas são comuns a todos. No painel (`python -m ragbench ui`) dá para abrir cada teste.",
             "* [DynaKRAG isolado](dynakrag/REPORT.md): Tabelas 1, 2, 6, 7, 8 e figuras",
             "* [RAG-Critic isolado](ragcritic/REPORT.md): Tabelas 3, 4, 5/7, 6, auditoria de formato e figuras",
             "* Planilhas (Excel): `metrics.csv` (todas as métricas por método × dataset, com o valor do artigo ao lado) "
             "e `comparacao.csv`", ""]
    for k, p in paths.items():
        if str(p).startswith("ERRO"):
            lines += [f"**Falha ao gerar o relatório {k}:**", "", "```", str(p), "```", ""]
    if rows:
        lines += ["## Estado dos jobs", "", "| tier | job | estado | min | erro |", "|---:|---|---|---:|---|", *rows, ""]
    total = sum(st.get("secs", 0) for st in status.values())
    lines.append(f"Tempo total de processamento registrado: {total / 3600:.2f} h.")
    path = os.path.join(res, "REPORT.md")
    os.makedirs(res, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    paths["index"] = path
    return paths
