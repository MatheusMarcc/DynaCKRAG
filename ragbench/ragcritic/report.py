"""Relatório do RAG-Critic isolado: RAG-Error Bench (+ auditoria de formato), Tabelas 3, 4 e 6, Fig. 6 e
diagnósticos do crítico, do planner e do executor."""
from __future__ import annotations

import glob
import os
from collections import Counter, defaultdict

from ..core.datasets import DATASET_NAMES
from ..core.io import read_json, read_jsonl
from ..core.report import PALETTE, fmt, md_table, plt, savefig
from ..core.runner import score
from ..core.stats import safe_mean
from ..paper_numbers import RAGCRITIC_ERRORBENCH, RAGCRITIC_FIG6, RAGCRITIC_TABLE3, RAGCRITIC_TABLE4
from .errorbench import load_metrics
from .taxonomy import TIER1, TIER2, canonical_tier1

PRETTY = {"standard_rag": "Standard RAG", "self_refine": "Self-Refine", "flare": "FLARE", "metarag": "MetaRAG",
          "ragcritic": "RAG-Critic", "ragcritic_n2": "RAG-Critic (N=2)",
          "ragcritic_zeroshot_critic": "w/o Critic Model (crítico = LLM base zero-shot)",
          "ragcritic_rule_planning": "w/o Auto-Planning (programa pela tabela T)"}
MAIN = ["standard_rag", "self_refine", "flare", "metarag", "ragcritic"]
ORDER = ["nq", "triviaqa", "hotpotqa", "2wiki", "toy_single", "toy_multihop"]


def load_runs(cfg: dict) -> dict:
    base = os.path.join(cfg["paths"]["results"], "ragcritic")
    runs: dict = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(base, "*", "*.jsonl"))):
        ds = os.path.basename(os.path.dirname(path))
        name = os.path.basename(path)[:-6]
        if ds in ("errorbench", "errorsystem") or "__" in name:
            continue
        rows = [r for r in read_jsonl(path) if not r.get("error")]
        if rows:
            runs[ds][name] = rows
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def _tokens(r: dict) -> float:
    t = r.get("tokens", 0) or 0
    g = r.get("gen_usage") or {}
    return t + (g.get("tokens", 0) or 0)


def _calls(r: dict) -> float:
    return (r.get("llm_calls", 0) or 0) + ((r.get("gen_usage") or {}).get("llm_calls", 0) or 0)


def _m(rows, key):
    return safe_mean([r.get(key) for r in rows if isinstance(r.get(key), (int, float))])


# ============================================================================ RAG-Error Bench
def section_errorbench(cfg: dict, fig_dir: str) -> str:
    mets = load_metrics(cfg)
    if not mets:
        return "_RAG-Error Bench ainda não rodou._\n"
    rows = []
    for key in sorted(mets):
        m = mets[key]
        if "identification" not in m:
            continue
        idf, cl = m["identification"], m["classification"]
        label = key.replace("__", " · ")
        rows.append([label, m.get("n"), fmt(idf["correct"], pct=True), fmt(idf["error"], pct=True), fmt(idf["avg"], pct=True),
                     fmt(cl["tag1_f1"], pct=True), fmt(cl["tag2_f1"], pct=True), fmt(cl["avg_f1"], pct=True),
                     fmt(cl["tag1_acc"], pct=True), fmt(cl["tag2_acc"], pct=True), m.get("parse_fail", 0)])
    out = md_table(["Modelo · condição", "n", "Ident. Correct", "Ident. Error", "Ident. Avg", "Tag1 F1", "Tag2 F1",
                    "Avg F1", "Tag1 acc", "Tag2 acc", "falhas de parse"], rows) + "\n\n"
    out += "<details><summary>Referência — Tabelas 5/7 do artigo (Correct / Error / Avg / Tag1 / Tag2 / Avg)</summary>\n\n"
    out += md_table(["Modelo", "Correct", "Error", "Avg", "Tag1", "Tag2", "Avg"],
                    [[k] + [fmt(x, 1) for x in v] for k, v in RAGCRITIC_ERRORBENCH.items()]) + "\n\n</details>\n\n"
    uni = mets.get("rag-critic-3b__uniform_indent")
    orig = mets.get("rag-critic-3b__original")
    swap = mets.get("rag-critic-3b__swapped")
    art = read_json(os.path.join(cfg["paths"]["results"], "ragcritic", "errorbench", "format_artifact.json"), {})
    out += "### Auditoria de formato do benchmark\n\n"
    if art:
        out += ("No arquivo oficial, o recuo das linhas do prompt (20 ou 16 espaços) separa as classes: "
                + "; ".join(f"`{k}`: " + ", ".join(f"{a}={b}" for a, b in sorted(v.items())) for k, v in art.items())
                + ". Ou seja, o formato sozinho já revela o rótulo.\n\n")
    if orig and swap:
        arow = [["prompts originais", fmt(orig["identification"]["correct"], pct=True), fmt(orig["identification"]["error"], pct=True),
                 fmt(orig["identification"]["avg"], pct=True)],
                ["recuo trocado (mesmas palavras)", fmt(swap["identification"]["correct"], pct=True),
                 fmt(swap["identification"]["error"], pct=True), fmt(swap["identification"]["avg"], pct=True)]]
        for k in ("all_indent20", "all_indent16"):
            if uni and k in uni:
                arow.append([f"todos com {k[-2:]} espaços", fmt(uni[k]["correct"], pct=True), fmt(uni[k]["error"], pct=True),
                             fmt(uni[k]["avg"], pct=True)])
        out += md_table(["RAG-Critic-3B — condição", "acerta 'Correct'", "acerta 'Error'", "média"], arow) + "\n\n"
        drop = orig["identification"]["avg"] - swap["identification"]["avg"]
        verdict = ("**o crítico está usando o atalho de formatação** (a identificação cai muito só com a troca do recuo)"
                   if drop > 0.25 else ("o crítico é parcialmente sensível ao formato" if drop > 0.08 else
                                        "o crítico não depende do recuo (a identificação quase não muda)"))
        out += f"Queda da identificação média com o recuo trocado: {100 * drop:+.1f} pontos → {verdict}.\n\n"
    # categorias (Fig. 7)
    if orig and "official" in orig:
        cats = orig["official"]["category_metrics"]["tag2"]
        fig, ax = plt().subplots(figsize=(7, 4.2))
        names = [c for c in TIER2]
        vals = [100 * cats.get(c, {}).get("f1", 0) for c in names]
        ax.barh(names, vals, color=PALETTE[0])
        ax.invert_yaxis()
        ax.set_xlabel("F1 por categoria de nível 2 (%)")
        ax.set_title("RAG-Critic-3B — prompts originais")
        ax.tick_params(axis="y", labelsize=7)
        savefig(fig, os.path.join(fig_dir, "fig7_errorbench_tier2.png"))
        out += "![F1 por categoria](figs/fig7_errorbench_tier2.png)\n\n"
    return out


# ============================================================================ Tabela 3
def section_table3(runs: dict) -> str:
    if not runs:
        return "_Nenhum resultado ainda._\n"
    datasets = list(runs)
    methods = [m for m in MAIN if any(m in runs[d] for d in datasets)]
    rows = []
    std = {d: {r["id"]: r for r in runs[d].get("standard_rag", [])} for d in datasets}
    for m in methods:
        row, f1s = [PRETTY[m]], []
        for d in datasets:
            rs = runs[d].get(m)
            if not rs:
                row += ["—", "—"]
                continue
            em, f1 = _m(rs, "em"), _m(rs, "f1")
            f1s.append(f1)
            delta = ""
            if m != "standard_rag" and std[d]:
                ids = [r["id"] for r in rs if r["id"] in std[d]]
                base = safe_mean([std[d][i]["f1"] for i in ids])
                delta = f" ({100 * (safe_mean([r['f1'] for r in rs if r['id'] in std[d]]) - base):+.1f})"
            row += [fmt(100 * em, 1), fmt(100 * f1, 1) + delta]
        overall = safe_mean(f1s)
        row.append(fmt(100 * overall, 1))
        row.append(fmt(safe_mean([_tokens(r) for d in datasets for r in runs[d].get(m, [])]), 0))
        row.append(fmt(safe_mean([_calls(r) for d in datasets for r in runs[d].get(m, [])]), 2))
        rows.append(row)
    headers = ["Método"] + [x for d in datasets for x in (f"{DATASET_NAMES.get(d, d)} EM", f"{DATASET_NAMES.get(d, d)} F1")] + \
              ["Overall F1", "Tokens/perg.", "Chamadas LLM"]
    out = md_table(headers, rows) + "\n\nEM e F1 em %; entre parênteses, a variação do F1 em relação ao Standard RAG nas mesmas perguntas.\n\n"
    prow = []
    for (bb, meth), v in RAGCRITIC_TABLE3.items():
        prow.append([f"{meth} ({bb})"] + [fmt(v[d][1], 1) if d in v else "—" for d in ("nq", "triviaqa", "hotpotqa", "2wiki")]
                    + [fmt(v.get("overall_f1"), 1)])
    out += "<details><summary>Referência — Tabela 3 do artigo (F1)</summary>\n\n" + md_table(
        ["Método", "NQ", "TriviaQA", "HotpotQA", "2Wiki", "Overall (7 datasets)"], prow) + "\n\n</details>\n\n"
    return out


# ============================================================================ Tabela 4 + 6
def section_ablations(runs: dict) -> str:
    datasets = list(runs)
    variants = ["ragcritic", "ragcritic_zeroshot_critic", "ragcritic_rule_planning", "standard_rag"]
    rows = []
    for v in variants:
        if not any(v in runs[d] for d in datasets):
            continue
        row = [PRETTY.get(v, v)]
        for d in datasets:
            rs = runs[d].get(v)
            row.append(fmt(100 * _m(rs, "f1"), 1) if rs else "—")
        flagged = [r for d in datasets for r in runs[d].get(v, []) if r.get("judgement_it1") == "Error"]
        n_all = sum(len(runs[d].get(v, [])) for d in datasets)
        row.append(f"{len(flagged)}/{n_all}" if v != "standard_rag" else "—")
        rows.append(row)
    out = md_table(["Variante"] + [f"{DATASET_NAMES.get(d, d)} F1" for d in datasets] + ["marcadas como erro"], rows)
    out += "\n\n<details><summary>Referência — Tabela 4 do artigo (Llama3.1-8B, F1)</summary>\n\n" + md_table(
        ["Variante", "NQ", "TriviaQA", "HotpotQA"], [[k] + [fmt(v[d], 1) for d in ("nq", "triviaqa", "hotpotqa")]
                                                    for k, v in RAGCRITIC_TABLE4.items()]) + "\n\n</details>\n\n"
    out += ("As linhas 'w/o Data-driven' e 'w/o Manual Sum.' do artigo exigem retreinar o crítico com outra taxonomia "
            "(fora do alcance de uma GPU de 4 GB); os scripts de treino estão em `ragcritic/training`.\n\n")
    # Tabela 6: iterações
    irows = []
    for d in datasets:
        s, n1, n2 = runs[d].get("standard_rag"), runs[d].get("ragcritic"), runs[d].get("ragcritic_n2")
        if s and n1:
            f0, f1_ = _m(s, "f1"), _m(n1, "f1")
            f2 = _m(n2, "f1") if n2 else float("nan")
            irows.append([DATASET_NAMES.get(d, d), fmt(100 * f0, 1), f"{100 * f1_:.1f} ({100 * (f1_ - f0):+.1f})",
                          (f"{100 * f2:.1f} ({100 * (f2 - f1_):+.1f})" if n2 else "—")])
    if irows:
        out += "### Tabela 6 — iterações do fluxo\n\n" + md_table(["Dataset", "N=0 (Standard RAG)", "+1 iteração", "+2 iterações"], irows) + "\n\n"
    return out


# ============================================================================ diagnóstico do crítico e do reparo
def section_critic_diag(cfg: dict, runs: dict, fig_dir: str) -> str:
    out = ""
    rows = []
    for v in ("ragcritic", "ragcritic_zeroshot_critic"):
        tp = fp = fn = tn = 0
        for d in runs:
            for r in runs[d].get(v, []):
                wrong = score(r, r.get("initial_pred", ""))["em"] < 1.0
                flag = r.get("judgement_it1") == "Error"
                tp += flag and wrong
                fp += flag and not wrong
                fn += (not flag) and wrong
                tn += (not flag) and not wrong
        n = tp + fp + fn + tn
        if not n:
            continue
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        rows.append([PRETTY.get(v, v), n, tp, fp, fn, tn, fmt(prec, 3), fmt(rec, 3), fmt((tp + tn) / n, 3)])
    if rows:
        out += ("Concordância do crítico com o EM nas respostas do Standard RAG (EM=0 tratado como erro; paráfrases "
                "corretas contam como 'erro', então a precisão real tende a ser maior):\n\n")
        out += md_table(["Crítico", "n", "erro & marcou", "certo & marcou", "erro & não marcou", "certo & não marcou",
                         "precisão", "revocação", "acurácia"], rows) + "\n\n"
    # transições do reparo
    trows = []
    for v in ("ragcritic", "ragcritic_rule_planning", "ragcritic_zeroshot_critic"):
        c = Counter()
        for d in runs:
            for r in runs[d].get(v, []):
                if r.get("judgement_it1") != "Error":
                    continue
                before = score(r, r.get("initial_pred", ""))["em"] == 1.0
                after = r.get("em", 0.0) == 1.0
                c[("certo" if before else "errado") + "→" + ("certo" if after else "errado")] += 1
        if c:
            trows.append([PRETTY.get(v, v), sum(c.values()), c["errado→certo"], c["certo→errado"], c["errado→errado"],
                          c["certo→certo"], f"{c['errado→certo'] - c['certo→errado']:+d}"])
    if trows:
        out += "Efeito do reparo nas respostas marcadas como erro (pelo EM):\n\n" + md_table(
            ["Variante", "marcadas", "errado→certo", "certo→errado", "errado→errado", "certo→certo", "saldo"], trows) + "\n\n"
    # executor
    base = os.path.join(cfg["paths"]["results"], "ragcritic")
    ex_rows, fn_count = [], Counter()
    for path in sorted(glob.glob(os.path.join(base, "*", "exec__*__it1.jsonl"))):
        rs = read_jsonl(path)
        if not rs:
            continue
        ok = safe_mean([float(bool(r.get("exec_ok"))) for r in rs])
        calls = safe_mean([len(r.get("calls") or []) for r in rs])
        for r in rs:
            fn_count.update(r.get("calls") or [])
        errs = Counter((r.get("exec_error") or "").split(":")[0] for r in rs if r.get("exec_error"))
        ex_rows.append([os.path.basename(os.path.dirname(path)), os.path.basename(path)[6:-12], len(rs), fmt(ok, 3),
                        fmt(calls, 2), ", ".join(f"{k} ({v})" for k, v in errs.most_common(3)) or "—"])
    if ex_rows:
        out += "Executor (iteração 1):\n\n" + md_table(["Dataset", "configuração", "programas", "executou sem erro",
                                                         "funções por programa", "erros mais comuns"], ex_rows) + "\n\n"
        out += "Uso das funções nos programas: " + ", ".join(f"{k}: {v}" for k, v in fn_count.most_common()) + "\n\n"
    # Fig. 6: distribuição de erros de nível 1 apontados pelo crítico
    t1 = Counter()
    for path in glob.glob(os.path.join(base, "*", "critic__finetuned*__it1.jsonl")):
        for r in read_jsonl(path):
            if r.get("judgement") == "Error":
                for t in r.get("tag1") or []:
                    c = canonical_tier1(t)
                    if c:
                        t1[c] += 1
    if t1:
        tot = sum(t1.values())
        fig, ax = plt().subplots(figsize=(6, 2.8))
        names = [x for x in TIER1 if x in t1]
        ax.barh(names, [t1[n] for n in names], color=PALETTE[2])
        for i, n in enumerate(names):
            ax.text(t1[n], i, f" {t1[n]} ({100 * t1[n] / tot:.1f}%)", va="center", fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("ocorrências (tags de nível 1 nas respostas marcadas como erro)")
        savefig(fig, os.path.join(fig_dir, "fig6_error_types.png"))
        stage = Counter("retrieval" if TIER1.index(n) < 3 else "generation" for n in t1.elements())
        out += (f"![Figura 6](figs/fig6_error_types.png)\n\nErros de retrieval: {100 * stage['retrieval'] / tot:.1f}% · "
                f"geração: {100 * stage['generation'] / tot:.1f}% (artigo: 41,3% × 58,7%). "
                "Artigo: " + ", ".join(f"{k} {v}%" for k, v in RAGCRITIC_FIG6.items()) + "\n\n")
    return out or "_Rode o fluxo principal primeiro._\n"


def section_errorsystem(cfg: dict) -> str:
    base = os.path.join(cfg["paths"]["results"], "ragcritic", "errorsystem")
    summ = read_json(os.path.join(base, "summary.json"), None)
    if not summ:
        return "_Etapas 1–3 (opcionais) ainda não rodaram: `python -m ragbench ragcritic errorsystem`._\n"
    tax = read_json(os.path.join(base, "taxonomy.json"), []) or []
    rows = [[c["cluster"], c["tier2_name"], c["size"], c["closest_official_tier2"], c["suggested_tier1"]] for c in tax]
    out = (f"{summ.get('n_samples')} respostas com erro, {summ.get('n_labels')} rótulos atômicos, {summ.get('n_clusters')} clusters. "
           f"Taxa de erro por modelo do pool: {summ.get('error_rate_by_model')}\n\n")
    return out + md_table(["Cluster", "Nome (LLM)", "rótulos", "nível 2 oficial mais próximo", "nível 1 sugerido"], rows) + "\n\n"


def build(cfg: dict) -> str:
    runs = load_runs(cfg)
    out_dir = os.path.join(cfg["paths"]["results"], "ragcritic")
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    m, rc = cfg["models"], cfg["ragcritic"]
    md = [f"# RAG-Critic isolado — relatório ({cfg['profile']})\n",
          f"Gerador π_α `{m['generator']}` · crítico π_θ `{m['critic']}` (recuo {rc['critic_template']}) · planner π_β "
          f"`{m['planner']}` · retriever `{cfg['retrieval']['ragcritic'].get('model', cfg['retrieval']['ragcritic']['kind'])}` top-{rc['top_k']} · "
          f"{rc['n_eval']} perguntas por dataset.\n",
          "> Compare tendências: o artigo usa geradores de 7B–70B, planner de 70B e o corpus wiki-2018 completo.\n",
          "## RAG-Error Bench (Tabelas 5 e 7)\n", section_errorbench(cfg, fig_dir),
          "## Tabela 3 — desempenho nos datasets de RAG\n", section_table3(runs),
          "## Tabela 4 — ablações\n", section_ablations(runs),
          "## Diagnóstico do crítico, do planner e do executor\n", section_critic_diag(cfg, runs, fig_dir),
          "## Sistema de erros construído localmente (etapas 1–3)\n", section_errorsystem(cfg)]
    text = "\n".join(md)
    path = os.path.join(out_dir, "REPORT.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path
