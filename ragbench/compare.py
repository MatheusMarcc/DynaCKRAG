"""Comparação direta com os artigos: ``COMPARACAO.md`` + ``metrics.csv`` + ``comparacao.csv``.

Os valores absolutos não devem coincidir com os dos artigos quando o modelo, o corpus ou a amostra são outros. O
que se compara são os EFEITOS: "o método X é melhor que Y", "tirar o componente Z piora". Para cada afirmação dos
artigos o mesmo Δ é calculado nas mesmas perguntas e recebe um veredito:

  mesmo sinal       o Δ da reprodução tem o sinal do Δ do artigo
  sinal oposto      o Δ da reprodução tem o sinal contrário
  sem diferença     o Δ da reprodução é zero
  sem referência    o artigo não traz número para essa comparação
  falta rodar       algum dos dois métodos ainda não terminou
"""
from __future__ import annotations

import csv
import math
import os

from .config import paper_datasets
from .core.datasets import DATASET_NAMES
from .core.io import read_jsonl
from .core.stats import safe_mean
from .paper_numbers import (DYNAKRAG_CAP_SWEEP, DYNAKRAG_TABLE1_QWEN7B, DYNAKRAG_TABLE2, RAGCRITIC_ERRORBENCH,
                            RAGCRITIC_ITERATIONS, RAGCRITIC_TABLE3, RAGCRITIC_TABLE4)

DK_BASELINES = ["ircot", "self_ask", "crag", "adaptive_rag", "s2g_rag", "corag_s2", "corag_s3", "par2_rag"]
NAMES = {"fixed_k": "Fixed-K", "ircot": "IRCoT", "self_ask": "Self-Ask+Search", "crag": "CRAG", "adaptive_rag": "Adaptive-RAG",
         "s2g_rag": "S2G-RAG", "corag_s2": "CoRAG-s2", "corag_s3": "CoRAG-s3", "par2_rag": "PAR²-RAG", "dynakrag": "DynaKRAG",
         "dynakrag_uniform": "DynaKRAG sem controlador aprendido", "dynakrag_no_sufficiency": "DynaKRAG sem sonda",
         "dynakrag_no_compression": "DynaKRAG sem compressão", "standard_rag": "Standard RAG", "ragcritic": "RAG-Critic",
         "ragcritic_n2": "RAG-Critic (2 iterações)", "self_refine": "Self-Refine", "flare": "FLARE", "metarag": "MetaRAG",
         "ragcritic_zeroshot_critic": "RAG-Critic com crítico zero-shot", "ragcritic_rule_planning": "RAG-Critic com plano por regra"}


# ============================================================================ leitura
def _rows(cfg: dict, paper: str, ds: str, method: str) -> dict:
    path = os.path.join(cfg["paths"]["results"], paper, ds, f"{method}.jsonl")
    return {r["id"]: r for r in read_jsonl(path) if not r.get("error")}


def _best_fixed_k(cfg: dict, ds: str) -> tuple[str | None, dict]:
    best, rows_best, f_best = None, {}, -1.0
    for k in cfg["dynakrag"]["baselines"]["fixed_k_grid"]:
        rows = _rows(cfg, "dynakrag", ds, f"fixed_k{k}")
        if rows:
            f = safe_mean([r["f1"] for r in rows.values()])
            if f > f_best:
                best, rows_best, f_best = f"fixed_k{k}", rows, f
    return best, rows_best


def _tokens(r: dict) -> float:
    return (r.get("tokens") or 0) + ((r.get("gen_usage") or {}).get("tokens") or 0)


def _label(name: str | None) -> str:
    if not name:
        return "—"
    if name.startswith("fixed_k") and name[7:].isdigit():
        return f"Fixed-K (K={name[7:]})"
    if name.startswith("dynakrag_cap") and name[12:].isdigit():
        return f"DynaKRAG (limite {name[12:]})"
    return NAMES.get(name, name)


# ============================================================================ veredito
def verdict(delta: float, paper_delta: float | None) -> str:
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return "falta rodar"
    if paper_delta is None or paper_delta == 0:
        return "sem referência"
    if delta == 0:
        return "sem diferença"
    return "mesmo sinal" if (delta > 0) == (paper_delta > 0) else "sinal oposto"


def paired(a: dict, b: dict, key: str = "f1", fn=None) -> dict:
    """Médias de A e de B e a diferença A − B, só nas perguntas respondidas pelos dois."""
    ids = sorted(set(a) & set(b))
    if not ids:
        return {"delta": float("nan"), "n": 0, "a": float("nan"), "b": float("nan")}
    get = fn or (lambda r: r.get(key, 0.0) or 0.0)
    xa, xb = [get(a[i]) for i in ids], [get(b[i]) for i in ids]
    return {"delta": safe_mean([x - y for x, y in zip(xa, xb)]), "n": len(ids), "a": safe_mean(xa), "b": safe_mean(xb),
            "wins": sum(x > y for x, y in zip(xa, xb)), "losses": sum(x < y for x, y in zip(xa, xb))}


# ============================================================================ afirmações dos artigos
def _pd(fn):
    """Δ do artigo, ou None quando o dataset não aparece na tabela do artigo (ex.: datasets de teste)."""
    try:
        return fn()
    except (KeyError, TypeError):
        return None


def _claim(paper: str, ds: str, claim: str, a_name: str, b_name: str | None, a_rows: dict, b_rows: dict,
           paper_delta, key: str = "f1", unit: str = "pts", fn=None) -> dict:
    nan = float("nan")
    r = paired(a_rows, b_rows, key, fn) if (a_rows and b_rows) else {"delta": nan, "n": 0, "a": nan, "b": nan}
    scale = 100 if unit == "pts" else 1
    ok = r["n"] > 0
    return {"paper": paper, "dataset": ds, "claim": claim, "a": _label(a_name), "b": _label(b_name),
            "metric": key, "unit": unit, "ours_a": r["a"] * scale if ok else nan, "ours_b": r["b"] * scale if ok else nan,
            "ours_delta": r["delta"] * scale if ok else nan, "n": r["n"],
            "paper_delta": paper_delta, "verdict": verdict(r["delta"] if ok else nan, paper_delta)}


def dynakrag_claims(cfg: dict) -> list[dict]:
    out = []
    t1, t2, cap = DYNAKRAG_TABLE1_QWEN7B, DYNAKRAG_TABLE2, DYNAKRAG_CAP_SWEEP
    for ds in paper_datasets(cfg, "dynakrag"):
        rows = lambda m: _rows(cfg, "dynakrag", ds, m)  # noqa: E731
        dk = rows("dynakrag")
        fk_name, fk = _best_fixed_k(cfg, ds)
        out.append(_claim("DynaKRAG", ds, "DynaKRAG supera o Fixed-K (melhor K)", "dynakrag", fk_name or "fixed_k", dk, fk,
                          _pd(lambda: 100 * (t1["DynaKRAG"][ds][1] - t1["Fixed-K"][ds][1]))))
        # baseline mais forte (entre os que já rodaram) × o mais forte do artigo
        best_b, best_rows, best_f = None, {}, -1.0
        for m in DK_BASELINES:
            rs = rows(m)
            if rs and safe_mean([r["f1"] for r in rs.values()]) > best_f:
                best_b, best_rows, best_f = m, rs, safe_mean([r["f1"] for r in rs.values()])
        out.append(_claim("DynaKRAG", ds, "DynaKRAG supera o baseline mais forte", "dynakrag", best_b, dk, best_rows,
                          _pd(lambda: 100 * (t1["DynaKRAG"][ds][1] - max(v[ds][1] for k, v in t1.items()
                                                                           if k not in ("DynaKRAG", "Fixed-K"))))))
        for claim, m, row in (("controle aprendido > escolha uniforme", "dynakrag_uniform", "w/o learned controller"),
                              ("a sonda de suficiência ajuda", "dynakrag_no_sufficiency", "w/o sufficiency check"),
                              ("a compressão terminal ajuda", "dynakrag_no_compression", "w/o terminal compression")):
            out.append(_claim("DynaKRAG", ds, claim, "dynakrag", m, dk, rows(m),
                              _pd(lambda row=row: 100 * (t2["DynaKRAG"][ds] - t2[row][ds]))))
        # eficiência: Δ em tokens por pergunta (negativo = DynaKRAG economiza)
        out.append(_claim("DynaKRAG", ds, "DynaKRAG gasta menos tokens que o S2G-RAG", "dynakrag", "s2g_rag", dk, rows("s2g_rag"),
                          _pd(lambda: t1["DynaKRAG"][ds][2] - t1["S2G-RAG"][ds][2]), key="tokens", unit="tokens", fn=_tokens))
        # varredura do limite de retrieval (Tabela 6)
        if ds == "musique":
            out.append(_claim("DynaKRAG", ds, "no MuSiQue, limite 2 > limite 3", "dynakrag_cap2", "dynakrag_cap3",
                              rows("dynakrag_cap2"), rows("dynakrag_cap3"), _pd(lambda: 100 * (cap[ds][2][1] - cap[ds][3][1]))))
        else:
            out.append(_claim("DynaKRAG", ds, "mais retrieval ajuda (limite 3 > limite 1)", "dynakrag_cap3", "dynakrag_cap1",
                              rows("dynakrag_cap3"), rows("dynakrag_cap1"), _pd(lambda: 100 * (cap[ds][3][1] - cap[ds][1][1]))))
    return out


def ragcritic_claims(cfg: dict) -> list[dict]:
    out = []
    t3, t4 = RAGCRITIC_TABLE3, RAGCRITIC_TABLE4
    for ds in paper_datasets(cfg, "ragcritic"):
        rows = lambda m: _rows(cfg, "ragcritic", ds, m)  # noqa: E731
        rc = rows("ragcritic")
        out.append(_claim("RAG-Critic", ds, "RAG-Critic supera o Standard RAG", "ragcritic", "standard_rag", rc, rows("standard_rag"),
                          _pd(lambda: t3[("Qwen2.5-7B", "RAG-Critic")][ds][1] - t3[("Qwen2.5-7B", "Standard RAG")][ds][1])))
        for m, key in (("self_refine", "Self-Refine"), ("flare", "FLARE"), ("metarag", "MetaRAG")):
            out.append(_claim("RAG-Critic", ds, f"RAG-Critic supera o {key}", "ragcritic", m, rc, rows(m),
                              _pd(lambda key=key: t3[("Llama3.1-8B", "RAG-Critic")][ds][1] - t3[("Llama3.1-8B", key)][ds][1])))
        out.append(_claim("RAG-Critic", ds, "o crítico ajustado ajuda (vs crítico zero-shot)", "ragcritic", "ragcritic_zeroshot_critic",
                          rc, rows("ragcritic_zeroshot_critic"), _pd(lambda: t4["RAG-Critic"][ds] - t4["w/o Critic Model"][ds])))
        out.append(_claim("RAG-Critic", ds, "o planejamento automático ajuda (vs plano por regra)", "ragcritic",
                          "ragcritic_rule_planning", rc, rows("ragcritic_rule_planning"),
                          _pd(lambda: t4["RAG-Critic"][ds] - t4["w/o Auto-Planning"][ds])))
        it = RAGCRITIC_ITERATIONS["Llama3-8B"]
        out.append(_claim("RAG-Critic", ds, "uma 2ª iteração ainda ajuda", "ragcritic_n2", "ragcritic", rows("ragcritic_n2"), rc,
                          _pd(lambda: round(it[2][ds] - it[1][ds], 2))))
    return out


def errorbench_rows(cfg: dict) -> list[dict]:
    """Identificação/classificação no RAG-Error Bench (sem teste pareado: arquivos de prompt diferentes)."""
    from .ragcritic.errorbench import load_metrics

    mets = load_metrics(cfg)
    out = []
    nan = float("nan")
    for key, m in sorted(mets.items()):
        tag, _, cond = key.partition("__")
        if cond == "uniform_indent":   # derivado: todos os prompts com o mesmo recuo (só identificação)
            for sub, idf in sorted(m.items()):
                out.append({"modelo": tag, "condicao": f"todos com recuo {sub[-2:]}", "n": None,
                            "correct": 100 * idf["correct"], "error": 100 * idf["error"], "avg": 100 * idf["avg"],
                            "tag1_f1": nan, "tag2_f1": nan, "paper_ref": None, "paper": None})
            continue
        if "identification" not in m:
            continue
        idf, cl = m["identification"], m["classification"]
        ref = None
        if tag == "rag-critic-3b" and cond == "original":
            ref = "RAG-Critic (3B)"
        elif tag.startswith("zeroshot") and cond == "original":
            ref = "Qwen2.5-7B"   # o artigo não tem 3B zero-shot; o Qwen2.5-7B é o mais próximo
        n = m.get("n") or len(read_jsonl(os.path.join(cfg["paths"]["results"], "ragcritic", "errorbench", f"{key}.jsonl")))
        out.append({"modelo": tag, "condicao": {"original": "prompt original", "swapped": "recuo trocado"}.get(cond, cond),
                    "n": n, "correct": 100 * idf["correct"], "error": 100 * idf["error"],
                    "avg": 100 * idf["avg"], "tag1_f1": 100 * cl["tag1_f1"], "tag2_f1": 100 * cl["tag2_f1"],
                    "paper_ref": ref, "paper": RAGCRITIC_ERRORBENCH.get(ref) if ref else None})
    return out


def format_audit(cfg: dict, tag: str = "rag-critic-3b") -> list[dict]:
    """Mesmo exemplo, recuo original × trocado: a decisão do crítico muda?"""
    d = os.path.join(cfg["paths"]["results"], "ragcritic", "errorbench")
    orig = {r["id"]: r for r in read_jsonl(os.path.join(d, f"{tag}__original.jsonl")) if not r.get("error")}
    swap = {r["id"]: r for r in read_jsonl(os.path.join(d, f"{tag}__swapped.jsonl")) if not r.get("error")}
    ids = sorted(set(orig) & set(swap))
    out = []
    for cls in ("Correct", "Error", None):
        sel = [i for i in ids if cls is None or orig[i]["label"] == cls]
        if not sel:
            continue
        a = [float(orig[i]["judgement"] == orig[i]["label"]) for i in sel]
        b = [float(swap[i]["judgement"] == swap[i]["label"]) for i in sel]
        flips = sum(orig[i]["judgement"] != swap[i]["judgement"] for i in sel)
        out.append({"classe": cls or "todas", "n": len(sel), "acc_original": 100 * safe_mean(a),
                    "acc_trocado": 100 * safe_mean(b), "delta": 100 * (safe_mean(a) - safe_mean(b)),
                    "decisoes_mudaram": 100 * flips / len(sel)})
    return out


# ============================================================================ tabela de métricas (CSV)
def metrics_table(cfg: dict) -> list[dict]:
    rows = []
    t1 = DYNAKRAG_TABLE1_QWEN7B
    paper_key = {"ircot": "IRCoT", "self_ask": "Self-Ask+Search", "crag": "CRAG", "adaptive_rag": "Adaptive-RAG",
                 "s2g_rag": "S2G-RAG", "corag_s3": "CoRAG-s3", "par2_rag": "PAR²-RAG", "dynakrag": "DynaKRAG"}
    for paper in ("dynakrag", "ragcritic"):
        base = os.path.join(cfg["paths"]["results"], paper)
        if not os.path.isdir(base):
            continue
        for ds in sorted(os.listdir(base)):
            d = os.path.join(base, ds)
            if not os.path.isdir(d) or ds in ("errorbench", "errorsystem", "figs"):
                continue
            for fn in sorted(os.listdir(d)):
                if not fn.endswith(".jsonl") or "__" in fn:
                    continue
                m = fn[:-6]
                rs = [r for r in read_jsonl(os.path.join(d, fn)) if not r.get("error")]
                if not rs:
                    continue
                row = {"artigo": paper, "dataset": ds, "metodo": m, "n": len(rs),
                       "EM": round(safe_mean([r.get("em", 0) for r in rs]), 4),
                       "F1": round(safe_mean([r.get("f1", 0) for r in rs]), 4),
                       "tokens_por_pergunta": round(safe_mean([_tokens(r) for r in rs]), 1),
                       "chamadas_llm": round(safe_mean([(r.get("llm_calls") or 0) + ((r.get("gen_usage") or {}).get("llm_calls") or 0)
                                                        for r in rs]), 2),
                       "chamadas_retrieval": round(safe_mean([r.get("n_ret", 0) or 0 for r in rs]), 2) if paper != "ragcritic" else "",
                       "SR_recall_suporte": (round(safe_mean([r["sr"] for r in rs if r.get("sr") is not None]), 4)
                                             if any(r.get("sr") is not None for r in rs) else ""),
                       "artigo_EM": "", "artigo_F1": "", "artigo_tokens": ""}
                if paper == "dynakrag" and m in paper_key and ds in t1.get(paper_key[m], {}):
                    em, f1, tok = t1[paper_key[m]][ds]
                    row.update(artigo_EM=em, artigo_F1=f1, artigo_tokens=tok)
                if paper == "dynakrag" and m.startswith("fixed_k") and ds in t1["Fixed-K"]:
                    em, f1, tok = t1["Fixed-K"][ds]
                    row.update(artigo_EM=em, artigo_F1=f1, artigo_tokens=tok)
                if paper == "ragcritic":
                    key = {"standard_rag": ("Qwen2.5-7B", "Standard RAG"), "ragcritic": ("Qwen2.5-7B", "RAG-Critic"),
                           "self_refine": ("Llama3.1-8B", "Self-Refine"), "flare": ("Llama3.1-8B", "FLARE"),
                           "metarag": ("Llama3.1-8B", "MetaRAG")}.get(m)
                    if key and ds in RAGCRITIC_TABLE3[key]:
                        row.update(artigo_EM=RAGCRITIC_TABLE3[key][ds][0] / 100, artigo_F1=RAGCRITIC_TABLE3[key][ds][1] / 100)
                rows.append(row)
    return rows


# ============================================================================ saída
def _fmt(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def _num(x, nd=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def _cell(v):
    """Número com vírgula decimal (Excel em português); NaN vira célula vazia."""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.4f}".rstrip("0").rstrip(".").replace(".", ",") if abs(v) < 1e6 else str(round(v))
    return v


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:   # utf-8-sig: o Excel abre com acentos
        w = csv.DictWriter(f, fieldnames=keys, delimiter=";")
        w.writeheader()
        for r in rows:
            w.writerow({k: _cell(v) for k, v in r.items()})


def build(cfg: dict) -> str:
    res = cfg["paths"]["results"]
    dk, rc, eb = dynakrag_claims(cfg), ragcritic_claims(cfg), errorbench_rows(cfg)
    lines = [f"# Comparação com os artigos ({cfg['profile']})", "",
             "Cada linha é uma afirmação de um dos artigos, testada nas MESMAS perguntas com os dois métodos.",
             "**Δ reprodução** é a diferença medida (em pontos de F1 ×100, ou em tokens); **Δ artigo** é a mesma "
             "diferença calculada a partir das tabelas do artigo. Os valores absolutos só são comparáveis quando o "
             "modelo, o corpus e a amostra são os do artigo; fora disso, compara-se o sinal da diferença.", "",
             "Vereditos: **mesmo sinal** · **sinal oposto** · **sem diferença** (Δ da reprodução igual a zero) · "
             "**sem referência** (o artigo não traz o número) · **falta rodar**.", ""]
    for title, claims in (("DynaKRAG", dk), ("RAG-Critic", rc)):
        lines += [f"## {title}", ""]
        if title == "DynaKRAG":
            from .dynakrag import bench as dkb
            from .dynakrag.inference import VARIANTS

            stale = sorted({DATASET_NAMES.get(ds, ds) for ds in paper_datasets(cfg, "dynakrag") for m in VARIANTS
                            if dkb.stale_control(cfg, ds, m)})
            if stale:
                lines += ["> **Atenção.** Os resultados do DynaKRAG de " + ", ".join(stale) + " foram gerados com as regras de "
                          "controle antigas, que não reproduzem a trajetória do artigo (ver `dynakrag/REPORT.md`). As linhas "
                          "abaixo que envolvem o DynaKRAG não valem como teste das afirmações do artigo até ele rodar de novo.", ""]
        lines += ["| Afirmação | Dataset | A | B | A (reprodução) | B (reprodução) | Δ reprodução | Δ artigo | Veredito |",
                  "|---|---|---|---|---:|---:|---:|---:|---|"]
        for c in claims:
            nd = 0 if c["unit"] == "tokens" else 1
            lines.append(f"| {c['claim']} | {DATASET_NAMES.get(c['dataset'], c['dataset'])} | {c['a']} | {c['b']} | "
                         f"{_num(c['ours_a'], nd)} | {_num(c['ours_b'], nd)} | {_fmt(c['ours_delta'], nd)} | "
                         f"{_fmt(c['paper_delta'], nd) if c['paper_delta'] is not None else '—'} | "
                         f"**{c['verdict']}** |")
        order = ("mesmo sinal", "sinal oposto", "sem diferença", "sem referência", "falta rodar")
        cnt = {v: sum(c["verdict"] == v for c in claims) for v in order}
        lines += ["", f"Resumo ({len(claims)} afirmações): " + " · ".join(f"{cnt[v]} {v}" for v in order) + ".", ""]
    lines += ["## RAG-Error Bench — o crítico identifica erros?", "",
              "Identificação = % de acerto do julgamento Correct/Error (métrica oficial `caculate_acc.py`); Tag1/Tag2 F1 = "
              "acerto do tipo de erro (nível 1 = 7 categorias, nível 2 = 19).", "",
              "| Modelo | Condição | n | acerta 'Correct' | acerta 'Error' | média | Tag1 F1 | Tag2 F1 | artigo: média / Tag1 / Tag2 |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in eb:
        p = r["paper"]
        ref = f"{p[2]} / {p[3]} / {p[4]} ({r['paper_ref']})" if p else "—"
        lines.append(f"| {r['modelo']} | {r['condicao']} | {r['n'] if r['n'] else '—'} | {_num(r['correct'])} | {_num(r['error'])} | "
                     f"{_num(r['avg'])} | {_num(r['tag1_f1'])} | {_num(r['tag2_f1'])} | {ref} |")
    if not eb:
        lines.append("| falta rodar | | | | | | | | |")
    audit = format_audit(cfg)
    lines += ["", "### Auditoria de formato (o crítico lê o conteúdo ou o recuo?)", "",
              "No benchmark oficial todo exemplo 'Correct' tem recuo de 20 espaços e todo 'Error' tem 16. Aqui o MESMO "
              "exemplo é mostrado com o recuo original e com o recuo trocado (as palavras são idênticas). Um crítico que "
              "lê o conteúdo dá a mesma resposta nos dois casos; uma queda grande indica o atalho. Esta auditoria não "
              "consta no artigo.", ""]
    if audit:
        lines += ["| Classe | n | acerto (original) | acerto (recuo trocado) | queda (original − trocado) | decisões que mudaram |",
                  "|---|---:|---:|---:|---:|---:|"]
        for a in audit:
            lines.append(f"| {a['classe']} | {a['n']} | {_num(a['acc_original'])} | {_num(a['acc_trocado'])} | "
                         f"{_fmt(a['delta'])} | {_num(a['decisoes_mudaram'])}% |")
        tot = audit[-1]
        verdict_fmt = ("o crítico depende do formato" if tot["delta"] >= 10 else
                       "efeito pequeno do formato" if tot["delta"] > 0 else "sem dependência do formato")
        lines += ["", f"Leitura: **{verdict_fmt}** (queda de {tot['delta']:.1f} pontos; "
                      f"{tot['decisoes_mudaram']:.0f}% das decisões mudaram).", ""]
    else:
        lines += ["Falta rodar (job `ragcritic/errorbench/rag-critic-3b`, tier 1).", ""]
    lines += ["Arquivos para planilha (abrem direto no Excel): `metrics.csv` (todas as métricas por método e dataset, "
              "com os valores do artigo ao lado) e `comparacao.csv` (as tabelas de afirmações acima).", "",
              "## Glossário das métricas", "",
              "* **EM** (exact match): 1 se a resposta, normalizada (minúsculas, sem artigos e pontuação), é igual a alguma "
              "resposta de referência; média nas perguntas.",
              "* **F1**: sobreposição de palavras entre resposta e referência (a melhor entre as referências). É a métrica "
              "principal dos dois artigos; nas tabelas aparece ×100 (pontos).",
              "* **Tokens por pergunta**: soma dos tokens de todos os prompts enviados ao LLM para responder uma pergunta "
              "(o custo que o DynaKRAG usa na Tabela 1).",
              "* **Chamadas ao LLM / ao retrieval**: quantas vezes cada método chamou o modelo e o buscador por pergunta.",
              "* **SR** (support recall): fração dos parágrafos de suporte do gabarito que o método recuperou. No DynaKRAG é "
              "o alvo que o controlador aprende a aumentar (Eq. 1); na análise de falhas separa falha de busca de falha de "
              "raciocínio.",
              "* **Identificação** (RAG-Error Bench): % de acerto do julgamento Correct/Error em cada classe e a média das "
              "duas. **Tag1/Tag2 F1**: F1 da categoria de erro prevista (níveis 1 e 2 da taxonomia).", ""]
    path = os.path.join(res, "COMPARACAO.md")
    os.makedirs(res, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    write_csv(os.path.join(res, "metrics.csv"), metrics_table(cfg))
    write_csv(os.path.join(res, "comparacao.csv"), [
        {"artigo": c["paper"], "dataset": c["dataset"], "afirmacao": c["claim"], "A": c["a"], "B": c["b"],
         "A_reproducao": c["ours_a"], "B_reproducao": c["ours_b"], "delta_reproducao": c["ours_delta"], "n": c["n"],
         "delta_artigo": c["paper_delta"] if c["paper_delta"] is not None else "", "veredito": c["verdict"]}
        for c in dk + rc])
    return path
