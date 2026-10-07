"""Análise de falhas por teste: onde cada método falha e se os métodos falham nos mesmos testes.

Um **teste** é uma pergunta do dataset, com o seu gabarito, respondida por um método. Esta análise junta as
respostas de todos os métodos para a mesma pergunta e responde, na ordem em que o painel mostra:

1. **Testagem padrão** — as métricas de cada método (EM, F1, custo).
2. **Falhas** — os testes em que o método falhou pelo critério de ``compare.fail_metric`` (padrão: EM = 0), cada
   um com a ETAPA em que a falha aparece, medida contra o gabarito:

   * ``formato``    — a resposta certa está lá, mas não na forma pedida (o gabarito aparece dentro da resposta,
     ou a resposta é um pedaço do gabarito);
   * ``busca``      — faltou evidência: nem todos os parágrafos de suporte do gabarito foram recuperados;
   * ``raciocinio`` — todos os parágrafos de suporte foram recuperados e mesmo assim a resposta saiu errada;
   * ``indefinido`` — o dataset não anota os parágrafos de suporte e a resposta não é localizável por texto.

3. **Falhas em comum** — os testes em que TODOS os métodos escolhidos falham, e a sobreposição das falhas
   par a par (contagens e índice de Jaccard).

A etapa usa o gabarito e é um diagnóstico deste pacote; nenhum dos dois artigos a define. As mesmas regras
estão no painel (``ui/static/app.js``), que recalcula tudo no navegador quando o critério muda; o teste
``tests/test_analysis.py`` confere que os dois lados dão os mesmos números.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import time
from collections import Counter, defaultdict

from .core.datasets import DATASET_NAMES, dataset_dir
from .core.io import read_jsonl
from .core.report import md_table
from .core.text import best_em_f1, contains_answer, norm_title, normalize_answer, support_recall

PAPER_ORDER = ("ragcritic", "dynakrag")
PAPER_LABEL = {"ragcritic": "RAG-Critic", "dynakrag": "DynaKRAG"}
SKIP_DIRS = {"errorbench", "errorsystem", "figs"}
STAGES = ("busca", "raciocinio", "formato", "indefinido")
STAGE_LABEL = {"ok": "Acertou", "busca": "Busca", "raciocinio": "Raciocínio", "formato": "Formato",
               "indefinido": "Indefinido"}
FAMILY_ORDER = ("principal", "base", "baseline", "ablacao", "variante", "outro")
FAMILY_LABEL = {"principal": "Métodos principais", "base": "Pontos de partida (bases simples)",
                "baseline": "Baselines dos artigos", "ablacao": "Ablações", "variante": "Variantes e diagnósticos",
                "outro": "Outros"}
BEST_FIXED_K = "dynakrag/@melhor_fixed_k"       # resolvido por dataset: o Fixed-K de maior F1 médio


# ============================================================================ registro dos métodos
def _m(label, family, desc, base=None, short=None):
    return {"label": label, "short": short or label, "family": family, "desc": desc, "base": base}


_STD = "ragcritic/standard_rag"
_RC = {
    "standard_rag": _m("Standard RAG", "base",
                       "Busca os 5 parágrafos mais parecidos com a pergunta e gera a resposta uma única vez. "
                       "É o ponto de partida do RAG-Critic."),
    "ragcritic": _m("RAG-Critic", "principal",
                    "Standard RAG + crítico (RAG-Critic-3B) que julga a resposta. Se julgar errada, um planner "
                    "escreve um programa de correção e o executor roda.", _STD),
    "ragcritic_n2": _m("RAG-Critic (N=2)", "variante", "RAG-Critic com 2 rodadas de crítica e correção.", _STD,
                       short="RC · N=2"),
    "ragcritic_zeroshot_critic": _m("RAG-Critic · crítico zero-shot", "ablacao",
                                    "Ablação: troca o crítico treinado por um LLM comum, sem treino.", _STD,
                                    short="RC · crítico zero-shot"),
    "ragcritic_rule_planning": _m("RAG-Critic · plano por regra", "ablacao",
                                  "Ablação: sem planner — o programa de correção vem de uma tabela fixa erro→ação.",
                                  _STD, short="RC · plano por regra"),
    "self_refine": _m("Self-Refine", "baseline", "Baseline do RAG-Critic: o próprio modelo critica e refina a resposta.",
                      _STD),
    "flare": _m("FLARE", "baseline",
                "Baseline do RAG-Critic: se o rascunho tem tokens de baixa confiança, busca de novo e regenera.", _STD),
    "metarag": _m("MetaRAG", "baseline",
                  "Baseline do RAG-Critic: compara com a resposta sem documentos, avalia e decide por regras.", _STD),
}
_DK = {
    "dynakrag": _m("DynaKRAG", "principal",
                   "Um controlador aprendido escolhe a cada passo: buscar mais, buscar a lacuna, reescrever a "
                   "pergunta, expandir a entidade-ponte, checar suficiência ou parar. No fim comprime a evidência "
                   "e responde.", BEST_FIXED_K),
    "dynakrag_no_compression": _m("DynaKRAG · sem compressão", "ablacao",
                                  "Ablação: sem a compressão final da evidência.", BEST_FIXED_K,
                                  short="DK · sem compressão"),
    "dynakrag_no_sufficiency": _m("DynaKRAG · sem sonda", "ablacao",
                                  "Ablação: sem a ação de checar se a evidência é suficiente.", BEST_FIXED_K,
                                  short="DK · sem sonda"),
    "dynakrag_uniform": _m("DynaKRAG · controlador ao acaso", "ablacao",
                           "Ablação: as ações são sorteadas, sem o controlador aprendido.", BEST_FIXED_K,
                           short="DK · ao acaso"),
    "ircot": _m("IRCoT", "baseline", "Baseline do DynaKRAG: intercala uma frase de raciocínio e uma busca.",
                BEST_FIXED_K),
    "self_ask": _m("Self-Ask + busca", "baseline",
                   "Baseline do DynaKRAG: faz perguntas de acompanhamento e busca cada uma.", BEST_FIXED_K,
                   short="Self-Ask"),
    "crag": _m("CRAG", "baseline",
               "Baseline do DynaKRAG: avalia os documentos (correto, ambíguo, incorreto) e refina ou busca de novo.",
               BEST_FIXED_K),
    "adaptive_rag": _m("Adaptive-RAG", "baseline",
                       "Baseline do DynaKRAG: classifica a complexidade da pergunta e escolhe a estratégia.",
                       BEST_FIXED_K),
    "s2g_rag": _m("S2G-RAG", "baseline",
                  "Baseline do DynaKRAG: julga a suficiência e transforma as lacunas em novas buscas.", BEST_FIXED_K),
    "corag_s2": _m("CoRAG (2 passos)", "baseline", "Baseline do DynaKRAG: cadeia de 2 sub-perguntas e sub-respostas.",
                   BEST_FIXED_K, short="CoRAG-s2"),
    "corag_s3": _m("CoRAG (3 passos)", "baseline", "Baseline do DynaKRAG: cadeia de 3 sub-perguntas e sub-respostas.",
                   BEST_FIXED_K, short="CoRAG-s3"),
    "par2_rag": _m("PAR²-RAG", "baseline",
                   "Baseline do DynaKRAG: cobre as sub-perguntas em largura e depois refina em profundidade.",
                   BEST_FIXED_K),
}


def method_info(key: str) -> dict:
    """Rótulo, família, base e descrição de um método (``artigo/nome``)."""
    paper, _, name = key.partition("/")
    info = None
    if paper == "ragcritic":
        info = _RC.get(name)
    elif paper == "dynakrag":
        info = _DK.get(name)
        if info is None and name.startswith("fixed_k") and name[7:].isdigit():
            k = name[7:]
            info = _m(f"Fixed-K (K={k})", "base",
                      f"Busca os {k} parágrafos mais parecidos com a pergunta e responde. Referência estática do "
                      "DynaKRAG.", short=f"Fixed-K {k}")
        elif info is None and name.startswith("dynakrag_cap") and name[12:].isdigit():
            c = name[12:]
            info = _m(f"DynaKRAG · até {c} busca(s)", "variante", f"Variante: no máximo {c} ação(ões) de busca.",
                      BEST_FIXED_K, short=f"DK · cap {c}")
    if info is None:
        info = _m(name, "outro", "")
    return {"key": key, "paper": paper, "name": name, **info}


def _order(key: str) -> tuple:
    """Ordem de exibição: artigo → base, principal, ablação, variante, baseline."""
    info = method_info(key)
    paper = PAPER_ORDER.index(info["paper"]) if info["paper"] in PAPER_ORDER else 9
    fam = {"base": 0, "principal": 1, "ablacao": 2, "variante": 3, "baseline": 4, "outro": 5}[info["family"]]
    m = re.search(r"(\d+)$", info["name"])
    return (paper, fam, info["name"].rstrip("0123456789"), int(m.group(1)) if m else 0)


# ============================================================================ leitura
def result_files(cfg: dict) -> dict:
    """{dataset: {'artigo/método': caminho}} — só os arquivos finais (os de estágio têm '__' no nome)."""
    res = cfg["paths"]["results"]
    out: dict = defaultdict(dict)
    for paper in PAPER_ORDER:
        for path in sorted(glob.glob(os.path.join(res, paper, "*", "*.jsonl"))):
            ds, name = os.path.basename(os.path.dirname(path)), os.path.basename(path)[:-6]
            if "__" in name or ds in SKIP_DIRS:
                continue
            out[ds][f"{paper}/{name}"] = path
    return {ds: dict(sorted(ms.items(), key=lambda kv: _order(kv[0]))) for ds, ms in out.items()}


def dataset_order(names) -> list[str]:
    pref = ["hotpotqa", "2wiki", "musique", "nq", "triviaqa", "toy_multihop", "toy_single"]
    return sorted(names, key=lambda d: (pref.index(d) if d in pref else 99, d))


def load_eval(cfg: dict, dataset: str) -> dict:
    """{id: registro} na ordem do ``eval.jsonl`` (vazio se os dados não estão nesta máquina)."""
    return {r["id"]: r for r in read_jsonl(os.path.join(dataset_dir(cfg, dataset), "eval.jsonl"))}


_CORPUS_CACHE: dict = {}


def corpus_index(cfg: dict, dataset: str) -> dict | None:
    """Corpus indexado por id e por título normalizado (para mostrar e checar os documentos recuperados)."""
    path = os.path.join(dataset_dir(cfg, dataset), "corpus.jsonl")
    if not os.path.exists(path):
        return None
    key = (path, os.path.getmtime(path))
    if key not in _CORPUS_CACHE:
        _CORPUS_CACHE.clear()                               # um corpus por vez na memória
        docs = read_jsonl(path)
        by_title: dict = defaultdict(list)
        for d in docs:
            by_title[norm_title(d.get("title", ""))].append(d)
        _CORPUS_CACHE[key] = {"by_id": {d["id"]: d for d in docs}, "by_title": dict(by_title), "norm": {}, "n": len(docs)}
    return _CORPUS_CACHE[key]


def title_of_contents(c: str) -> str:
    """Título de um documento no formato do FlashRAG / RAG-Critic: ``"Título"\\ntexto``."""
    first = (c or "").split("\n", 1)[0].strip()
    return first[1:-1] if len(first) >= 2 and first[0] == '"' and first[-1] == '"' else ""


def row_titles(row: dict, fallback: dict | None = None) -> list[str] | None:
    """Títulos de tudo o que o método recuperou. ``None`` = a linha não registra (resultado antigo)."""
    if row.get("doc_titles") is not None:
        return list(row["doc_titles"])
    if row.get("contents"):
        return [t for t in (title_of_contents(c) for c in row["contents"]) if t]
    if fallback is not None and not row.get("changed") and not row.get("rounds") and \
            row.get("meta_action") != "re-retrieve":
        # resultado antigo de um método que parte do Standard RAG e não buscou de novo: os mesmos documentos
        return row_titles(fallback)
    return None


def row_docs(row: dict, corpus: dict | None, fallback: dict | None = None) -> list[dict]:
    """Documentos recuperados, com texto: pelos ids, senão pelos títulos, senão pelo campo ``contents``."""
    out, seen = [], set()

    def add(d: dict):
        k = d.get("id") or (norm_title(d.get("title", "")), (d.get("text") or "")[:80])
        if k not in seen:
            seen.add(k)
            out.append(d)

    ids = row.get("doc_ids")
    if ids and corpus:
        for i in ids:
            if i in corpus["by_id"]:
                add(corpus["by_id"][i])
    elif corpus and row_titles(row, fallback):
        for t in row_titles(row, fallback) or []:
            for d in corpus["by_title"].get(norm_title(t), [])[:2]:
                add(d)
    if not out:
        src = row.get("contents") or (fallback or {}).get("contents") or []
        for c in src:
            title, _, text = (c or "").partition("\n")
            add({"id": None, "title": title.strip().strip('"'), "text": text.strip() or (c or "")})
    return out


# ============================================================================ a célula (um teste)
_ABST = re.compile(           # frases de quem não respondeu, em qualquer ponto da resposta
    r"\b(not (?:enough|sufficient)|insufficient (?:information|evidence|data|context)|"
    r"cannot (?:be )?(?:determined?|answer(?:ed)?|find|tell)|can(?:'|no)t (?:determine|answer|find|tell)|unable to|"
    r"no (?:information|answer|mention|evidence|data)\b|not (?:mentioned|provided|specified|stated|available|given|found|clear|known)|"
    r"do(?:es)? not (?:provide|mention|contain|specify|state|know|say)|don'?t know)", re.I)
_ABST_WHOLE = re.compile(r"^\W*(unknown|n/?a|none|no answer|not sure|insufficient evidence)\W*$", re.I)   # só se for a resposta inteira
_YESNO = {"yes", "no", "noanswer"}


def is_abstention(pred: str) -> bool:
    """Sem resposta: vazia, "unknown", ou do tipo "the documents do not mention…"."""
    p = (pred or "").strip()
    return not p or bool(_ABST_WHOLE.match(p)) or bool(_ABST.search(p))


def is_near(pred: str, golds: list[str]) -> bool:
    """A resposta é um pedaço contíguo do gabarito (ex.: "Obama" para "Barack Obama")."""
    p = normalize_answer(pred).split()
    if not p:
        return False
    for g in golds or []:
        gt = normalize_answer(g).split()
        if not gt or " ".join(gt) in _YESNO or len(p) >= len(gt):
            continue
        if any(gt[i:i + len(p)] == p for i in range(len(gt) - len(p) + 1)):
            return True
    return False


def answer_in_docs(golds: list[str], docs: list[dict], corpus: dict | None = None) -> bool | None:
    """O texto do gabarito aparece em algum documento recuperado? ``None`` quando não dá para dizer (sim/não)."""
    gs = [g for g in (normalize_answer(x) for x in golds or []) if g and g not in _YESNO]
    if not gs or not docs:
        return None if not gs else False
    cache = corpus["norm"] if corpus else {}
    for d in docs:
        k = d.get("id")
        text = cache.get(k) if k else None
        if text is None:
            text = normalize_answer(f"{d.get('title', '')} {d.get('text', '')}")
            if k:
                cache[k] = text
        if any(g in text for g in gs):
            return True
    return False


def make_cell(row: dict, rec: dict | None, corpus: dict | None, fallback: dict | None = None) -> dict:
    """Tudo o que o painel e os relatórios precisam de um teste, em chaves curtas (vai para o navegador)."""
    rec = rec or {}
    pred = str(row.get("pred") or "")
    golds = row.get("answers") or rec.get("answers") or []
    em, f1 = row.get("em"), row.get("f1")
    if em is None or f1 is None:
        em, f1 = best_em_f1(pred, golds)
    acc = row.get("acc")
    if acc is None:
        acc = contains_answer(pred, golds)
    sup = rec.get("supporting_titles") or []
    titles = row_titles(row, fallback)
    sr = support_recall(titles, sup) if (sup and titles is not None) else None
    docs = row_docs(row, corpus, fallback)
    return {
        "p": pred, "e": float(em), "f": round(float(f1), 4), "a": float(acc),
        "s": None if sr is None else round(sr, 4),          # recall dos parágrafos de suporte
        "i": answer_in_docs(golds, docs, corpus),           # o gabarito aparece nos documentos recuperados?
        "n": is_near(pred, golds),                          # a resposta é um pedaço do gabarito
        "b": is_abstention(pred),                           # sem resposta
        "x": bool(row.get("changed")),                      # o método mexeu na resposta da sua base
        "t": row.get("tokens"), "c": row.get("llm_calls"), "r": row.get("n_ret"),
    }


# ============================================================================ critério e etapa
def is_fail(c: dict, metric: str = "f1", thr: float = 0.5) -> bool:
    if metric == "em":
        return c["e"] < 1.0
    if metric == "acc":
        return c["a"] < 1.0
    return c["f"] < thr


def stage_of(c: dict, metric: str = "f1", thr: float = 0.5) -> str:
    """'ok' ou a etapa em que a falha aparece (ver o texto no topo do módulo)."""
    if not is_fail(c, metric, thr):
        return "ok"
    if c["a"] >= 1.0 or c["n"]:
        return "formato"
    cov = (c["s"] >= 1.0) if c["s"] is not None else c["i"]
    if cov is None:
        return "indefinido"
    return "raciocinio" if cov else "busca"


def criterion_text(metric: str, thr: float) -> str:
    if metric == "em":
        return "EM = 0 (a resposta não é idêntica ao gabarito depois de normalizar)"
    if metric == "acc":
        return "o gabarito não aparece dentro da resposta"
    return f"F1 < {str(thr).replace('.', ',')}"


# ============================================================================ a matriz testes × métodos
def load_matrix(cfg: dict, dataset: str) -> dict:
    """Todos os testes de um dataset: perguntas (linhas) × métodos (colunas)."""
    files = result_files(cfg).get(dataset, {})
    recs = load_eval(cfg, dataset)
    corpus = corpus_index(cfg, dataset)
    rows_by, errors = {}, {}
    for key, path in files.items():
        rows = read_jsonl(path)
        errors[key] = sum(bool(r.get("error")) for r in rows)
        rows_by[key] = {r["id"]: r for r in rows if not r.get("error") and "id" in r}
    seen = [i for i in recs if any(i in rows for rows in rows_by.values())]
    known = set(seen)
    for rows in rows_by.values():                   # perguntas que só existem nos resultados (dados ausentes)
        for i in rows:
            if i not in known:
                known.add(i)
                seen.append(i)
    std = rows_by.get(_STD, {})
    questions = []
    for i in seen:
        rec = recs.get(i) or next(rows[i] for rows in rows_by.values() if i in rows)
        questions.append({"id": i, "q": rec.get("question", ""), "gold": list(rec.get("answers") or []),
                          "type": rec.get("type") or "", "hops": rec.get("hops"),
                          "sup": list(rec.get("supporting_titles") or [])})
    cells, methods = {}, []
    for key, rows in rows_by.items():
        if not rows:
            continue
        fb = std if key.startswith("ragcritic/") and key != _STD else {}
        cells[key] = [make_cell(rows[i], recs.get(i), corpus, fb.get(i)) if i in rows else None for i in seen]
        methods.append({**method_info(key), "n": len(rows), "n_errors": errors[key],
                        "mtime": os.path.getmtime(files[key])})
    # a base dos métodos do DynaKRAG é o melhor Fixed-K deste dataset (como no artigo)
    best = best_fixed_k(cells)
    for m in methods:
        if m["base"] == BEST_FIXED_K:
            m["base"] = best
        elif m["base"] and m["base"] not in cells:
            m["base"] = None
    return {"dataset": dataset, "name": DATASET_NAMES.get(dataset, dataset), "profile": cfg.get("profile"),
            "questions": questions, "methods": methods, "cells": cells, "has_data": bool(recs),
            "has_corpus": corpus is not None, "has_support": any(q["sup"] for q in questions)}


def best_fixed_k(cells: dict) -> str | None:
    best, best_f1 = None, -1.0
    for key, col in cells.items():
        if key.startswith("dynakrag/fixed_k"):
            vals = [c["f"] for c in col if c]
            f1 = sum(vals) / len(vals) if vals else -1.0
            if f1 > best_f1 + 1e-12:
                best, best_f1 = key, f1
    return best


def default_methods(cfg: dict, available) -> list[str]:
    """Os métodos marcados de início: os de ``compare.default_methods`` que já têm resultado."""
    want = list((cfg.get("compare") or {}).get("default_methods") or [])
    return [k for k in want if k in available]


def main_methods(matrix: dict, selected: list[str]) -> list[str]:
    """Dos escolhidos, os métodos propriamente ditos (sem as bases simples) — é sobre eles que se cruza."""
    fam = {m["key"]: m["family"] for m in matrix["methods"]}
    return [k for k in selected if fam.get(k) != "base"]


# ============================================================================ contagens
def compute(matrix: dict, keys: list[str], metric: str = "f1", thr: float = 0.5) -> dict:
    """Estatísticas para um conjunto de métodos. O painel faz a mesma conta em JavaScript."""
    cells, qs = matrix["cells"], matrix["questions"]
    keys = [k for k in keys if k in cells]
    base_of = {m["key"]: m["base"] for m in matrix["methods"]}
    per = {}
    for k in keys:
        col = cells[k]
        idx = [i for i, c in enumerate(col) if c]
        st = Counter(stage_of(col[i], metric, thr) for i in idx)
        n = len(idx)
        fails = n - st["ok"]
        b = base_of.get(k)
        breaks = fixes = n_base = 0
        if b and b in cells:
            for i in idx:
                cb = cells[b][i]
                if not cb:
                    continue
                n_base += 1
                fb, fm = is_fail(cb, metric, thr), is_fail(col[i], metric, thr)
                breaks += (not fb) and fm
                fixes += fb and (not fm)
        per[k] = {"n": n, "em": _mean(col[i]["e"] for i in idx), "f1": _mean(col[i]["f"] for i in idx),
                  "fails": fails, "rate": fails / n if n else float("nan"),
                  "stages": {s: st[s] for s in STAGES}, "abst": sum(col[i]["b"] and is_fail(col[i], metric, thr) for i in idx),
                  "tokens": _mean(col[i]["t"] for i in idx if col[i]["t"] is not None),
                  "calls": _mean(col[i]["c"] for i in idx if col[i]["c"] is not None),
                  "base": b if (b and b in cells) else None, "n_base": n_base, "breaks": breaks, "fixes": fixes}
    common = [i for i in range(len(qs)) if all(cells[k][i] for k in keys)] if keys else []
    n = len(common)
    fail = {k: [is_fail(cells[k][i], metric, thr) for i in common] for k in keys}
    stg = {k: [stage_of(cells[k][i], metric, thr) for i in common] for k in keys}
    counts = [sum(fail[k][j] for k in keys) for j in range(n)]
    all_idx = [common[j] for j in range(n) if keys and counts[j] == len(keys)]
    pairs = []
    for x in range(len(keys)):
        for y in range(x + 1, len(keys)):
            a, b = keys[x], keys[y]
            na, nb = sum(fail[a]), sum(fail[b])
            both = sum(u and v for u, v in zip(fail[a], fail[b]))
            union = na + nb - both
            pairs.append({"a": a, "b": b, "fa": na, "fb": nb, "both": both,
                          "jaccard": both / union if union else float("nan")})
    same = Counter()
    for j in range(n):
        if keys and counts[j] == len(keys):
            ss = {stg[k][j] for k in keys}
            same[next(iter(ss)) if len(ss) == 1 else "misto"] += 1
    by_type: dict = defaultdict(lambda: {"n": 0, **{k: 0 for k in keys}})
    for j, i in enumerate(common):
        t = qs[i]["type"] or "—"
        by_type[t]["n"] += 1
        for k in keys:
            by_type[t][k] += fail[k][j]
    return {"keys": keys, "metric": metric, "thr": thr, "methods": per, "n": n, "common": common,
            "all_fail": {"n": len(all_idx), "idx": all_idx},
            "none_fail": sum(c == 0 for c in counts), "hist": [sum(c == k for c in counts) for k in range(len(keys) + 1)],
            "pairs": pairs, "same_stage": dict(same), "by_type": dict(by_type)}


def _mean(xs) -> float:
    xs = [float(x) for x in xs]
    return sum(xs) / len(xs) if xs else float("nan")


# ============================================================================ relatórios (FALHAS.md + CSV)
def _pct(x, d: int = 1) -> str:
    return "—" if x is None or x != x else f"{100 * x:.{d}f}".replace(".", ",")


def _num(x, d: int = 1) -> str:
    return "—" if x is None or x != x else f"{x:.{d}f}".replace(".", ",")


def _clip(s: str, n: int = 90) -> str:
    s = " ".join(str(s or "").split()).replace("|", "/")
    return s if len(s) <= n else s[: n - 1] + "…"


def section_dataset(cfg: dict, matrix: dict, metric: str, thr: float) -> tuple[str, dict]:
    label = {m["key"]: m["label"] for m in matrix["methods"]}
    all_keys = [m["key"] for m in matrix["methods"]]
    allst = compute(matrix, all_keys, metric, thr)
    out = f"## {matrix['name']}\n\n"
    if not matrix["has_support"]:
        out += ("_Este dataset não anota os parágrafos de suporte: a etapa da falha usa só o texto do gabarito "
                "(se ele aparece ou não nos documentos recuperados)._\n\n")
    # ------------------------------------------------------------------ 1 e 2
    out += "### 1 e 2 · Testagem padrão e falhas por método\n\n"
    paper_of = {m["key"]: m["paper"] for m in matrix["methods"]}
    for paper in PAPER_ORDER:
        rows = []
        for k in all_keys:
            if paper_of[k] != paper:
                continue
            s = allst["methods"][k]
            st = s["stages"]
            vs = "—" if not s["base"] else f"{s['breaks']} / {s['fixes']}"
            rows.append([label[k], s["n"], _pct(s["em"]), _pct(s["f1"]),
                         f"{s['fails']} ({_pct(s['rate'], 0)}%)", st["busca"], st["raciocinio"], st["formato"],
                         st["indefinido"], s["abst"], vs, label.get(s["base"], "—") if s["base"] else "—"])
        if rows:
            out += f"**{PAPER_LABEL[paper]}**\n\n" + md_table(
                ["Método", "testes", "EM", "F1", "falhas", "busca", "raciocínio", "formato", "indefinido", "sem resposta",
                 "quebras / consertos", "em relação a"], rows) + "\n\n"
    # ------------------------------------------------------------------ 3
    sel = main_methods(matrix, default_methods(cfg, set(all_keys)))
    stats = {"dataset": matrix["dataset"], "all": allst, "selected": sel, "common": None}
    if len(sel) < 2:
        have = ", ".join(label[k] for k in sel) or "nenhum"
        out += ("### 3 · Falhas em comum\n\n_Para cruzar as falhas é preciso ter pelo menos dois métodos principais "
                f"rodados neste dataset (por enquanto: {have}). Rode a comparação (`python -m ragbench bench`) e gere "
                "o relatório de novo; no painel dá para cruzar qualquer conjunto de métodos._\n\n")
        return out, stats
    c = compute(matrix, sel, metric, thr)
    stats["common"] = c
    af = c["all_fail"]
    names = ", ".join(label[k] for k in sel)
    out += f"### 3 · Falhas em comum — {names}\n\n"
    out += (f"Dos **{c['n']}** testes respondidos por todos eles, **{af['n']}** falham em todos "
            f"({_pct(af['n'] / c['n'] if c['n'] else float('nan'), 0)}%). Em {c['none_fail']} testes nenhum deles falha.\n\n")
    out += md_table(["Falharam em", *[str(k) for k in range(len(sel) + 1)]],
                    [["nº de testes", *c["hist"]]]) + "\n\n"
    if c["pairs"]:
        out += md_table(["Par de métodos", "falhas A", "falhas B", "falham os dois", "Jaccard"],
                        [[f"{label[p['a']]} × {label[p['b']]}", p["fa"], p["fb"], p["both"],
                          _num(p["jaccard"], 2)] for p in c["pairs"]]) + "\n\n"
    if af["n"]:
        ss = c["same_stage"]
        parts = [f"{ss[s]} na mesma etapa — {STAGE_LABEL[s].lower()}" for s in STAGES if ss.get(s)]
        if ss.get("misto"):
            parts.append(f"{ss['misto']} em etapas diferentes de um método para outro")
        out += "**Falham no mesmo elemento?** Das falhas em comum: " + "; ".join(parts) + ".\n\n"
        qs, cells = matrix["questions"], matrix["cells"]
        rows = []
        for i in af["idx"][:40]:
            q = qs[i]
            rows.append([f"`{q['id']}`", _clip(q["q"], 80), _clip(" / ".join(q["gold"][:2]), 40),
                         *[f"{_clip(cells[k][i]['p'] or '(vazio)', 30)} · {STAGE_LABEL[stage_of(cells[k][i], metric, thr)].lower()}"
                           for k in sel]])
        out += md_table(["id", "Pergunta", "Gabarito", *[label[k] for k in sel]], rows, "l" * (3 + len(sel))) + "\n\n"
        if af["n"] > 40:
            out += f"_Mostrando 40 de {af['n']}; a lista completa está em `falhas_em_comum.csv` e no painel._\n\n"
    if len(c["by_type"]) > 1:
        rows = [[t, v["n"], *[f"{v[k]} ({_pct(v[k] / v['n'], 0)}%)" for k in sel]]
                for t, v in sorted(c["by_type"].items(), key=lambda kv: -kv[1]["n"])]
        out += "**Falhas por tipo de pergunta**\n\n" + md_table(["Tipo", "testes", *[label[k] for k in sel]], rows) + "\n\n"
    return out, stats


def _write_csv(path: str, header: list[str], rows: list[list]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:      # utf-8-sig + ';' abre direto no Excel pt-BR
        w = csv.writer(f, delimiter=";")
        w.writerow(header)
        for r in rows:
            w.writerow([("" if v is None else (str(v).replace(".", ",") if isinstance(v, float) else v)) for v in r])
    return path


def write_reports(cfg: dict) -> dict:
    """Gera ``FALHAS.md`` e três planilhas com o que já terminou de rodar."""
    cmp_cfg = cfg.get("compare") or {}
    metric, thr = str(cmp_cfg.get("fail_metric", "em")), float(cmp_cfg.get("fail_threshold", 0.5))
    res = cfg["paths"]["results"]
    files = result_files(cfg)
    body, per_test, summary, common_rows = "", [], [], []
    for ds in dataset_order(files):
        matrix = load_matrix(cfg, ds)
        if not matrix["methods"]:
            continue
        text, st = section_dataset(cfg, matrix, metric, thr)
        body += text
        qs, cells = matrix["questions"], matrix["cells"]
        info = {m["key"]: m for m in matrix["methods"]}
        n_fail = [sum(1 for k in cells if cells[k][i] and is_fail(cells[k][i], metric, thr)) for i in range(len(qs))]
        n_ans = [sum(1 for k in cells if cells[k][i]) for i in range(len(qs))]
        for k, col in cells.items():
            s = st["all"]["methods"][k]
            summary.append([ds, info[k]["paper"], info[k]["name"], info[k]["label"], info[k]["family"],
                            s["n"], s["em"], s["f1"], s["fails"], s["rate"],
                            *[s["stages"][x] for x in STAGES], s["abst"], info[k]["base"] or "", s["breaks"], s["fixes"],
                            s["tokens"], s["calls"]])
            b = info[k]["base"]
            for i, c in enumerate(col):
                if not c:
                    continue
                cb = cells[b][i] if b and b in cells else None
                fm = is_fail(c, metric, thr)
                fb = is_fail(cb, metric, thr) if cb else None
                per_test.append([ds, qs[i]["id"], qs[i]["type"], qs[i]["q"], " | ".join(qs[i]["gold"]), info[k]["paper"],
                                 info[k]["name"], info[k]["label"], c["p"], c["e"], c["f"], c["a"], c["s"],
                                 "" if c["i"] is None else ("sim" if c["i"] else "não"), "sim" if fm else "não",
                                 stage_of(c, metric, thr), "sim" if c["b"] else "não",
                                 "sim" if (fb is False and fm) else "não", "sim" if (fb and not fm) else "não",
                                 n_fail[i], n_ans[i]])
        if st["common"]:
            sel = st["selected"]
            for i in st["common"]["all_fail"]["idx"]:
                row = [ds, qs[i]["id"], qs[i]["type"], qs[i]["q"], " | ".join(qs[i]["gold"])]
                for k in sel:
                    row += [info[k]["label"], cells[k][i]["p"], stage_of(cells[k][i], metric, thr)]
                common_rows.append(row)
    head = [f"# Análise de falhas — perfil {cfg.get('profile')}", "",
            f"Gerado em {time.strftime('%Y-%m-%d %H:%M')}. Um **teste** é uma pergunta do dataset respondida por um "
            f"método. **Critério de falha:** {criterion_text(metric, thr)} (mude em `compare.fail_metric` / "
            "`compare.fail_threshold`, ou no painel, onde o corte é ajustável na hora).", "",
            "**Etapa da falha** (medida contra o gabarito; é um diagnóstico deste pacote, não dos artigos):", "",
            "* **formato** — a resposta certa está lá, mas não na forma pedida (o gabarito aparece dentro da "
            "resposta, ou a resposta é um pedaço do gabarito);",
            "* **busca** — faltou evidência: nem todos os parágrafos de suporte do gabarito foram recuperados;",
            "* **raciocínio** — todos os parágrafos de suporte foram recuperados e a resposta saiu errada;",
            "* **indefinido** — sem parágrafos de suporte anotados e sem como localizar o gabarito nos documentos.", "",
            "**Quebras / consertos** — em relação ao ponto de partida do próprio método (Standard RAG para o "
            "RAG-Critic e as baselines dele; o melhor Fixed-K para o DynaKRAG): "
            "quebra = a base acerta e o método falha; conserto = a base falha e o método acerta.", "",
            "**Atenção ao comparar entre artigos:** cada artigo roda isolado, com o retriever do próprio artigo "
            "(RAG-Critic: E5-base-v2; DynaKRAG: BGE-large). Uma falha de busca num e não no outro pode "
            "vir do retriever, não do método. Para igualar: `--set retrieval.ragcritic.model=BAAI/bge-large-en-v1.5`.",
            "", "Para abrir cada teste (pergunta, gabarito, parágrafos de suporte, resposta e passo a passo de cada "
            "método): `python -m ragbench ui`.", ""]
    if not body:
        body = "_Nenhum resultado ainda. Rode `python -m ragbench bench` e gere o relatório de novo._\n"
    os.makedirs(res, exist_ok=True)
    md = os.path.join(res, "FALHAS.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(head) + "\n" + body)
    out = {"md": md}
    out["csv_testes"] = _write_csv(
        os.path.join(res, "falhas_por_teste.csv"),
        ["dataset", "id", "tipo", "pergunta", "gabarito", "artigo", "metodo", "rotulo", "resposta", "em", "f1", "acc",
         "recall_suporte", "gabarito_nos_documentos", "falhou", "etapa", "sem_resposta", "quebra", "conserto",
         "metodos_que_falharam", "metodos_que_responderam"], per_test)
    out["csv_resumo"] = _write_csv(
        os.path.join(res, "falhas_resumo.csv"),
        ["dataset", "artigo", "metodo", "rotulo", "familia", "testes", "em", "f1", "falhas",
         "taxa_de_falha", *STAGES, "sem_resposta", "base", "quebras", "consertos", "tokens", "chamadas_llm"], summary)
    width = max((len(r) for r in common_rows), default=5)
    extra = [x for j in range((width - 5) // 3) for x in (f"metodo_{j + 1}", f"resposta_{j + 1}", f"etapa_{j + 1}")]
    out["csv_comum"] = _write_csv(os.path.join(res, "falhas_em_comum.csv"),
                                  ["dataset", "id", "tipo", "pergunta", "gabarito", *extra], common_rows)
    return out


__all__ = ["result_files", "load_matrix", "compute", "is_fail", "stage_of", "method_info", "default_methods",
           "main_methods", "write_reports", "corpus_index", "load_eval", "row_docs", "row_titles"]
