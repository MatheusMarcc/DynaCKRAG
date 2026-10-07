"""Relatório do DynaKRAG isolado: tabelas no formato do artigo + figuras + testes pareados."""
from __future__ import annotations

import glob
import os
from collections import defaultdict

from ..core.datasets import DATASET_NAMES
from ..core.io import read_json, read_jsonl
from ..core.report import PALETTE, fmt, md_table, plt, savefig
from ..core.stats import safe_mean
from ..paper_numbers import (DYNAKRAG_ACTIONS, DYNAKRAG_CAP_SWEEP, DYNAKRAG_FIG5, DYNAKRAG_PROBE, DYNAKRAG_TABLE1_QWEN7B,
                             DYNAKRAG_TABLE2)
from .controller import controller_path
from .state import control_rules, control_signature, row_control

PRETTY = {"ircot": "IRCoT", "self_ask": "Self-Ask+Search", "crag": "CRAG", "adaptive_rag": "Adaptive-RAG",
          "s2g_rag": "S2G-RAG", "corag_s2": "CoRAG-s2", "corag_s3": "CoRAG-s3", "par2_rag": "PAR²-RAG",
          "dynakrag": "DynaKRAG", "dynakrag_no_compression": "w/o terminal compression",
          "dynakrag_no_sufficiency": "w/o sufficiency check", "dynakrag_uniform": "w/o learned controller"}
TABLE1_ORDER = ["Fixed-K", "IRCoT", "S2G-RAG", "CoRAG", "Adaptive-RAG", "PAR²-RAG", "CRAG", "Self-Ask+Search", "DynaKRAG"]


def load_runs(cfg: dict) -> dict:
    base = os.path.join(cfg["paths"]["results"], "dynakrag")
    runs: dict = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(base, "*", "*.jsonl"))):
        ds = os.path.basename(os.path.dirname(path))
        rows = [r for r in read_jsonl(path) if not r.get("error")]
        if rows:
            runs[ds][os.path.basename(path)[:-6]] = rows
    return runs


def _mean(rows, key):
    return safe_mean([r.get(key) for r in rows if isinstance(r.get(key), (int, float))])


def _pick_fixed_k(methods: dict) -> tuple[str | None, list]:
    cands = {m: rows for m, rows in methods.items() if m.startswith("fixed_k")}
    if not cands:
        return None, []
    best = max(cands, key=lambda m: _mean(cands[m], "f1"))
    return best, cands[best]


def _pick_corag(runs: dict) -> str | None:
    scores = {}
    for m in ("corag_s2", "corag_s3"):
        vals = [_mean(runs[ds][m], "f1") for ds in runs if m in runs[ds]]
        if vals:
            scores[m] = safe_mean(vals)
    return max(scores, key=scores.get) if scores else None


def table1_rows(runs: dict) -> tuple[list[str], dict]:
    datasets = [d for d in ("hotpotqa", "2wiki", "musique", "toy_multihop") if d in runs] + \
               [d for d in runs if d not in ("hotpotqa", "2wiki", "musique", "toy_multihop")]
    corag = _pick_corag(runs)
    table: dict = defaultdict(dict)
    for ds in datasets:
        methods = runs[ds]
        fk_name, fk_rows = _pick_fixed_k(methods)
        mapping = {"Fixed-K": fk_rows, "IRCoT": methods.get("ircot"), "S2G-RAG": methods.get("s2g_rag"),
                   "CoRAG": methods.get(corag) if corag else None, "Adaptive-RAG": methods.get("adaptive_rag"),
                   "PAR²-RAG": methods.get("par2_rag"), "CRAG": methods.get("crag"),
                   "Self-Ask+Search": methods.get("self_ask"), "DynaKRAG": methods.get("dynakrag")}
        for name, rows in mapping.items():
            if rows:
                table[name][ds] = {"em": _mean(rows, "em"), "f1": _mean(rows, "f1"), "tok": _mean(rows, "tokens"),
                                   "n": len(rows), "calls": _mean(rows, "llm_calls"), "ret": _mean(rows, "n_ret"),
                                   "secs": _mean(rows, "llm_seconds"), "sr": _mean(rows, "sr"),
                                   "note": (f"K={fk_name[7:]}" if name == "Fixed-K" and fk_name else
                                            (corag[-2:] if name == "CoRAG" and corag else ""))}
    return datasets, table


def _mark(values: list[float]) -> list[str]:
    """negrito = melhor, sublinhado = segundo melhor (como no artigo)."""
    clean = sorted({round(v, 6) for v in values if v == v}, reverse=True)
    out = []
    for v in values:
        s = fmt(v)
        if v == v and clean and round(v, 6) == clean[0]:
            s = f"**{s}**"
        elif v == v and len(clean) > 1 and round(v, 6) == clean[1]:
            s = f"<u>{s}</u>"
        out.append(s)
    return out


def section_table1(runs: dict) -> str:
    datasets, table = table1_rows(runs)
    if not table:
        return "_Nenhum resultado ainda._\n"
    names = [n for n in TABLE1_ORDER if n in table]
    headers = ["Método"] + [f"{DATASET_NAMES.get(d, d)} {c}" for d in datasets for c in ("EM↑", "F1↑", "Tok.↓")]
    cols = {}
    for d in datasets:
        for c, key in (("EM↑", "em"), ("F1↑", "f1")):
            cols[(d, key)] = _mark([table[n].get(d, {}).get(key, float("nan")) for n in names])
    rows = []
    for i, n in enumerate(names):
        notes = [table[n].get(d, {}).get("note", "") for d in datasets]
        if n == "Fixed-K" and len(set(notes)) > 1:      # o melhor K muda de dataset para dataset: mostra todos
            label = n + " (" + ", ".join(f"{DATASET_NAMES.get(d, d)} {x}" for d, x in zip(datasets, notes) if x) + ")"
        else:
            label = n + (f" ({notes[0]})" if notes and notes[0] else "")
        row = [label]
        for d in datasets:
            cell = table[n].get(d)
            if not cell:
                row += ["—", "—", "—"]
                continue
            row += [cols[(d, "em")][i], cols[(d, "f1")][i], fmt(cell["tok"], 1)]
        rows.append(row)
    out = md_table(headers, rows) + "\n\n"
    n_eval = {d: max((table[n].get(d, {}).get("n", 0) for n in names), default=0) for d in datasets}
    out += "n por dataset: " + ", ".join(f"{DATASET_NAMES.get(d, d)}={n_eval[d]}" for d in datasets) + ". "
    out += "Tok. = tokens totais por pergunta (prompt + saída, todas as chamadas). Fixed-K = melhor K da grade por dataset; CoRAG = o mais forte entre s2 e s3 no geral.\n\n"
    # números do artigo
    pd = [d for d in datasets if d in ("hotpotqa", "2wiki", "musique")]
    if pd:
        prow = []
        for n in TABLE1_ORDER:
            key = "CoRAG-s3" if n == "CoRAG" else n
            if key in DYNAKRAG_TABLE1_QWEN7B:
                v = DYNAKRAG_TABLE1_QWEN7B[key]
                prow.append([n] + [x for d in pd for x in (fmt(v[d][0]), fmt(v[d][1]), fmt(v[d][2], 1))])
        out += "<details><summary>Referência — Tabela 1 do artigo (Qwen2.5-7B, splits completos)</summary>\n\n"
        out += md_table(["Método"] + [f"{DATASET_NAMES[d]} {c}" for d in pd for c in ("EM", "F1", "Tok.")], prow)
        out += "\n\n</details>\n\n"
    return out


def section_table2(runs: dict) -> str:
    datasets = [d for d in runs if "dynakrag" in runs[d]]
    if not datasets:
        return "_Rode o DynaKRAG e as ablações primeiro._\n"
    variants = [("DynaKRAG", "dynakrag"), ("w/o terminal compression", "dynakrag_no_compression"),
                ("w/o sufficiency check", "dynakrag_no_sufficiency"), ("w/o learned controller", "dynakrag_uniform")]
    rows = []
    for label, m in variants + [("Fixed-K static reference", None)]:
        row = [label]
        for d in datasets:
            rs = runs[d].get(m) if m else _pick_fixed_k(runs[d])[1]
            cell = f"{_mean(rs, 'f1'):.4f}" if rs else "—"
            p = DYNAKRAG_TABLE2.get(label, {}).get(d)
            row += [cell, fmt(p) if p is not None else "—"]
        rows.append(row)
    headers = ["Variante"] + [x for d in datasets for x in (f"{DATASET_NAMES.get(d, d)} F1", "artigo")]
    out = md_table(headers, rows) + "\n\n"
    return out


def section_caps(runs: dict, fig_dir: str) -> str:
    datasets = [d for d in runs if any(m.startswith("dynakrag_cap") for m in runs[d])]
    if not datasets:
        return "_Varredura de cap ainda não rodou (tier 3)._\n"
    rows = []
    fig, axes = plt().subplots(1, len(datasets), figsize=(3.6 * len(datasets), 2.8), squeeze=False)
    for j, d in enumerate(datasets):
        xs, ys = [], []
        for c in (1, 2, 3):
            rs = runs[d].get(f"dynakrag_cap{c}")
            if not rs:
                continue
            ex = safe_mean([float(r.get("exhausted", False)) for r in rs])
            rows.append([DATASET_NAMES.get(d, d), c, fmt(_mean(rs, "em")), fmt(_mean(rs, "f1")), fmt(_mean(rs, "tokens"), 1),
                         fmt(_mean(rs, "n_ret"), 3), fmt(_mean(rs, "llm_calls"), 3), fmt(ex)])
            xs.append(c)
            ys.append(_mean(rs, "f1"))
        ax = axes[0][j]
        ax.plot(xs, ys, "o-", color=PALETTE[0], label="ragbench")
        if d in DYNAKRAG_CAP_SWEEP:
            pc = DYNAKRAG_CAP_SWEEP[d]
            ax.plot(list(pc), [v[1] for v in pc.values()], "s--", color=PALETTE[1], alpha=0.7, label="artigo")
        ax.set_title(DATASET_NAMES.get(d, d))
        ax.set_xlabel("limite de chamadas de retrieval")
        ax.set_xticks([1, 2, 3])
        if j == 0:
            ax.set_ylabel("F1")
            ax.legend(frameon=False)
    savefig(fig, os.path.join(fig_dir, "fig3_cap_sweep.png"))
    out = md_table(["Dataset", "cap", "EM", "F1", "Tokens", "Ret. calls", "LLM calls", "Esgotado"], rows)
    return out + "\n\n![Figura 3](figs/fig3_cap_sweep.png)\n\n"


def paper_early_stop(d: str) -> float | None:
    """Fração que para antes do limite no artigo, tirada da Tabela 6: 1 − Exhausted com limite de 3 retrievals."""
    cap3 = (DYNAKRAG_CAP_SWEEP.get(d) or {}).get(3)
    return 1 - cap3[5] if cap3 else None


def section_actions(runs: dict, fig_dir: str) -> str:
    datasets = [d for d in runs if "dynakrag" in runs[d]]
    if not datasets:
        return ""
    acts = ["retrieve_more", "sufficiency_check", "gap_query", "rewrite_query", "bridge_entity_expand"]
    rows, comp = [], {}
    for d in datasets:
        rs = runs[d]["dynakrag"]
        cnt = {a: safe_mean([r.get("actions", []).count(a) for r in rs]) for a in acts}
        compress = safe_mean([float(r.get("compression_raw") is not None) for r in rs])
        comp[d] = cnt
        p = DYNAKRAG_ACTIONS.get(d)
        rows.append([DATASET_NAMES.get(d, d)] + [fmt(cnt[a], 3) for a in acts] +
                    [fmt(compress, 3), fmt(_mean(rs, "n_ret"), 3), fmt(_mean(rs, "llm_calls"), 3), fmt(_mean(rs, "steps"), 3),
                     fmt(safe_mean([float(r.get("stopped_by") == "controller") for r in rs]), 3)])
        if p:
            pe = paper_early_stop(d)
            rows.append(["↳ artigo"] + [fmt(p[a], 3) for a in acts] + [fmt(p["compress"], 3), fmt(p["ret_calls"], 3),
                                                                        fmt(p["llm_calls"], 3),
                                                                        fmt(DYNAKRAG_PROBE[d][0], 3) if d in DYNAKRAG_PROBE else "—",
                                                                        fmt(pe, 3) if pe is not None else "—"])
    out = md_table(["Dataset", "Retrieve", "Suff.", "Gap", "Rewrite", "Bridge", "Compress", "Ret. calls", "LLM calls",
                    "Passos", "parou antes do limite"], rows)
    out += ("\n\nLinhas \"↳ artigo\": Tabela 7 (ações e chamadas), Tabela 8 (passos) e, na última coluna, 1 − *Exhausted* da "
            "Tabela 6 com limite de 3 retrievals (a fração que não usa as três buscas). No artigo a sonda roda uma vez em "
            "quase toda pergunta e a maioria segue até o limite; uma composição muito diferente aqui quer dizer que o "
            "controlador não está se comportando como o do artigo, e os números de acerto não valem como DynaKRAG.")
    # o que a sonda respondeu e o que aconteceu depois (só nas linhas que gravaram o veredito)
    vrows = []
    for d in datasets:
        rs = [r for r in runs[d]["dynakrag"] if "suff" in r]
        if not rs:
            continue
        n = len(rs)
        yes = [r for r in rs if r["suff"] == 1]
        no = [r for r in rs if r["suff"] == 0]
        never = [r for r in rs if r["suff"] == -1]
        def cell(xs):
            k = sum(1 for r in xs if r.get("stopped_by") == "controller")
            return f"{len(xs)} ({k} {'parou' if k == 1 else 'pararam'} antes do limite)"

        vrows.append([DATASET_NAMES.get(d, d), n, cell(yes), cell(no), cell(never),
                      sum(1 for r in rs if r.get("compression_fallback"))])
    if vrows:
        out += ("\n\n**Veredito da sonda e parada** (por pergunta):\n\n" +
                md_table(["Dataset", "n", "sonda: suficiente", "sonda: falta evidência", "sonda não rodou",
                          "compressão sem texto (respondeu com os documentos)"], vrows))
    fig, ax = plt().subplots(figsize=(6.5, 0.6 + 0.55 * len(datasets)))
    left = [0.0] * len(datasets)
    labels = {"retrieve_more": "Retrieve more", "sufficiency_check": "Sufficiency check", "gap_query": "Gap query",
              "rewrite_query": "Rewrite", "bridge_entity_expand": "Bridge"}
    for k, a in enumerate(acts):
        vals = [comp[d][a] for d in datasets]
        ax.barh([DATASET_NAMES.get(d, d) for d in datasets], vals, left=left, color=PALETTE[k], label=labels[a])
        left = [x + y for x, y in zip(left, vals)]
    ax.set_xlabel("média de ações escolhidas por exemplo")
    ax.legend(ncol=3, frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    savefig(fig, os.path.join(fig_dir, "fig4_action_composition.png"))
    return out + "\n\n![Figura 4](figs/fig4_action_composition.png)\n\n"


def section_probe(runs: dict) -> str:
    rows = []
    for d in runs:
        full, nop = runs[d].get("dynakrag"), runs[d].get("dynakrag_no_sufficiency")
        if not (full and nop):
            continue
        p = DYNAKRAG_PROBE.get(d)
        nop_stop = safe_mean([float(r.get("stopped_by") == "controller") for r in nop])
        rows.append([DATASET_NAMES.get(d, d), fmt(_mean(full, "steps"), 3), fmt(_mean(nop, "steps"), 3),
                     fmt(_mean(full, "f1")), fmt(_mean(nop, "f1")), fmt(nop_stop, 3),
                     " / ".join(fmt(x, 3) for x in p) if p else "—"])
    if not rows:
        return "_Precisa de `dynakrag` e `dynakrag_no_sufficiency`._\n"
    return md_table(["Dataset", "Passos (completo)", "Passos (sem sonda)", "F1 completo", "F1 sem sonda",
                     "sem sonda: parou antes do limite", "artigo (passos/passos/F1/F1)"], rows) + "\n\n"


def _type_key(d: str, r: dict) -> str:
    if d == "musique" and r.get("hops"):
        return f"{r['hops']}-hop"
    return r.get("type") or "?"


def section_structure(runs: dict, fig_dir: str) -> str:
    comps = [("Fixed-K", None), ("Uniform", "dynakrag_uniform"), ("No suff.", "dynakrag_no_sufficiency"),
             ("No comp.", "dynakrag_no_compression")]
    out = ""
    for d in runs:
        if "dynakrag" not in runs[d]:
            continue
        full = {r["id"]: r for r in runs[d]["dynakrag"]}
        types = sorted({_type_key(d, r) for r in full.values()})
        grid, rows = [], []
        for t in types:
            ids_t = [i for i, r in full.items() if _type_key(d, r) == t]
            line = []
            for label, m in comps:
                other_rows = _pick_fixed_k(runs[d])[1] if m is None else runs[d].get(m)
                if not other_rows:
                    line.append(float("nan"))
                    continue
                other = {r["id"]: r for r in other_rows}
                ids = [i for i in ids_t if i in other]
                line.append(100 * (safe_mean([full[i]["f1"] for i in ids]) - safe_mean([other[i]["f1"] for i in ids]))
                            if ids else float("nan"))
            grid.append(line)
            rows.append([t, len(ids_t)] + [f"{v:+.1f}" if v == v else "—" for v in line] +
                        ([" / ".join(f"{DYNAKRAG_FIG5[t][c]:+.1f}" for c, _ in comps)] if d == "2wiki" and t in DYNAKRAG_FIG5 else ["—"]))
        out += f"**{DATASET_NAMES.get(d, d)}** — ΔF1 (pontos) do DynaKRAG completo em relação a cada comparador:\n\n"
        out += md_table(["Tipo", "n", "vs Fixed-K", "vs Uniform", "vs No suff.", "vs No comp.", "artigo (2Wiki)"], rows) + "\n\n"
        try:
            import numpy as np

            arr = np.array(grid, dtype=float)
            fig, ax = plt().subplots(figsize=(4.8, 0.5 + 0.45 * len(types)))
            vmax = np.nanmax(np.abs(arr)) if np.isfinite(arr).any() else 1.0
            im = ax.imshow(arr, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(comps)), [c for c, _ in comps])
            ax.set_yticks(range(len(types)), types)
            for (i, j), v in np.ndenumerate(arr):
                if v == v:
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, label="ΔF1 (pontos)")
            ax.set_title(DATASET_NAMES.get(d, d))
            savefig(fig, os.path.join(fig_dir, f"fig5_structure_{d}.png"))
            out += f"![Figura 5 — {d}](figs/fig5_structure_{d}.png)\n\n"
        except Exception:  # noqa: BLE001
            pass
    return out or "_Precisa do DynaKRAG completo e das ablações._\n"


def section_controller(cfg: dict, runs: dict) -> str:
    rows, imps, notes = [], [], []
    for d in runs:
        rep = read_json(controller_path(cfg, d).replace(".pkl", "_report.json"), None)
        if not rep:
            continue
        v, c = rep.get("value", {}), rep.get("continuation", {})
        mae = "; ".join(f"{a}: {x['mae']:.3f}" for a, x in sorted(v.get("cv", {}).items()))
        cal = c.get("tau_calibration") or {}
        rows.append([DATASET_NAMES.get(d, d), rep.get("n_train"), v.get("n_pairs"), fmt(v.get("argmax_agreement"), 3),
                     mae, c.get("n", "—"), fmt(c.get("positive_rate"), 3), fmt(c.get("cv_auc"), 3), fmt(c.get("tau"), 2),
                     cal.get("n_can_stop", "—")])
        imps.append(f"- {DATASET_NAMES.get(d, d)}: " + ", ".join(f"{n} ({w:.3f})" for n, w in (v.get("importances") or [])[:8]))
        if cal.get("note"):
            notes.append(f"- {DATASET_NAMES.get(d, d)}: {cal['note']}")
    if not rows:
        return "_O controlador com as regras atuais ainda não foi treinado (rode o DynaKRAG)._\n"
    out = md_table(["Dataset", "perguntas de treino", "pares (s,a)", "acerto do argmax (CV)", "MAE por ação (CV)",
                    "prefixos (continuação)", "taxa 'continuar ajuda'", "AUC continuação (CV)", "τ",
                    "prefixos em que parar é permitido"], rows)
    if notes:
        out += "\n\nCalibração de τ:\n" + "\n".join(notes)
    return out + "\n\nFeatures mais importantes do modelo de valor:\n" + "\n".join(imps) + "\n\n"


def section_control(cfg: dict, runs: dict) -> str:
    """Regras de controle em vigor e aviso sobre resultados gerados com outras regras."""
    r = control_rules(cfg)
    sig = control_signature(cfg)
    stop = {"sufficient": "só depois que a sonda disse \"suficiente\" (ou no limite de passos)",
            "probe": "depois de qualquer veredito da sonda", "always": "sempre válida (regra das versões antigas)"}
    out = (f"Regras de controle: parada {stop.get(r['stop_rule'], r['stop_rule'])}; "
           f"{'sem limite de sondas' if r['max_probes'] is None else str(r['max_probes']) + ' sonda(s)'} por trajetória; sonda em "
           f"{'uma chamada' if r['probe'] == 'one_step' else 'duas chamadas'}. De onde vêm: `docs/DynaKRAG.md`, decisões 2 e 5.\n")
    stale = []
    for d in runs:
        for m, rs in runs[d].items():
            if m == "dynakrag" or m.startswith("dynakrag_"):
                other = sorted({row_control(x) for x in rs} - {sig})
                if other:
                    stale.append(f"{DATASET_NAMES.get(d, d)}/{m} ({'; '.join(other)})")
    if stale:
        out += ("\n> **Atenção — resultados de outras regras de controle.** " + ", ".join(stale) + ". Esses números foram "
                "gerados antes da correção da regra de parada e não reproduzem a trajetória do artigo (compare a composição das "
                "ações com a Tabela 7 mais abaixo). Rode o DynaKRAG de novo: os arquivos antigos são guardados em "
                "`ragbench_results/logs/controle_anterior/` e as variantes são recalculadas.\n")
    return out


def section_sr(runs: dict) -> str:
    rows = []
    for d in runs:
        for m in sorted(runs[d]):
            rs = runs[d][m]
            if any("sr" in r for r in rs):
                rows.append([DATASET_NAMES.get(d, d), PRETTY.get(m, m), fmt(_mean(rs, "sr"), 3), fmt(_mean(rs, "f1")),
                             fmt(_mean(rs, "llm_calls"), 2), fmt(_mean(rs, "n_ret"), 2), fmt(_mean(rs, "llm_seconds"), 1)])
    return md_table(["Dataset", "Método", "SR (recall de suporte)", "F1", "chamadas LLM", "chamadas retrieval",
                     "s de LLM/pergunta"], rows) + "\n\n" if rows else ""


def build(cfg: dict) -> str:
    runs = load_runs(cfg)
    out_dir = os.path.join(cfg["paths"]["results"], "dynakrag")
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    m = cfg["models"]
    md = [f"# DynaKRAG isolado — relatório ({cfg['profile']})\n",
          f"Gerador `{m['generator']}` · retriever `{cfg['retrieval']['dynakrag'].get('model', cfg['retrieval']['dynakrag']['kind'])}` · "
          f"{cfg['dynakrag']['n_eval']} perguntas de avaliação e {cfg['dynakrag']['n_train']} de treino por dataset · "
          f"RF 300 árvores / folha 8 / semente 13 · λ = {cfg['dynakrag']['lambda_cost']} · limiar de continuação "
          f"τ = {cfg['dynakrag']['controller']['continuation_threshold']} (auto = calibrado no treino; ver diagnóstico).\n",
          section_control(cfg, runs),
          "> Compare tendências, não valores absolutos: o artigo usa Qwen2.5-7B, o fullwiki e os splits completos.\n",
          "## Tabela 1 — acurácia e tokens\n", section_table1(runs),
          "## Tabela 2 — componentes (F1)\n", section_table2(runs),
          "## Figura 3 / Tabela 6 — limite de chamadas de retrieval\n", section_caps(runs, fig_dir),
          "## Figura 4 / Tabela 7 — composição das ações\n", section_actions(runs, fig_dir),
          "## Tabela 8 — diagnóstico da sonda de suficiência\n", section_probe(runs),
          "## Figura 5 — desempenho por estrutura de pergunta\n", section_structure(runs, fig_dir),
          "## Controlador aprendido (diagnóstico do treino)\n", section_controller(cfg, runs),
          "## Recall de suporte e custo por método\n", section_sr(runs)]
    text = "\n".join(md)
    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


__all__ = ["build", "load_runs", "table1_rows"]
