"""Análise de falhas: critério, etapa da falha, coincidência entre métodos e os arquivos gerados.

O painel refaz as mesmas contas em JavaScript; ``test_js_matches_python`` roda ``ui/static/app.js`` no Node
(quando o Node existe na máquina) e confere que os dois lados dão exatamente os mesmos números.
"""
import csv
import json
import math
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragbench import analysis as A  # noqa: E402
from ragbench.config import load_config  # noqa: E402

APP_JS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ragbench", "ui", "static", "app.js")


def cell(f=0.0, e=None, a=0.0, s=None, i=None, n=False, b=False, p="x"):
    return {"p": p, "e": float(f >= 1.0 if e is None else e), "f": f, "a": a, "s": s, "i": i, "n": n, "b": b, "x": False,
            "t": 100, "c": 1, "r": 1}


# ============================================================================ critério e etapa
def test_fail_criteria():
    c = cell(f=0.6, e=0.0, a=0.0)
    assert not A.is_fail(c, "f1", 0.5) and A.is_fail(c, "f1", 0.7) and A.is_fail(c, "em") and A.is_fail(c, "acc")
    assert not A.is_fail(cell(f=0.5), "f1", 0.5)                      # o corte é "abaixo de": F1 = 0,5 passa
    assert not A.is_fail(cell(f=0.2, a=1.0), "acc")


def test_stage_rules():
    assert A.stage_of(cell(f=1.0)) == "ok"
    assert A.stage_of(cell(f=0.3, a=1.0, s=0.0)) == "formato"         # o gabarito está dentro da resposta: formato vence
    assert A.stage_of(cell(f=0.4, n=True, s=1.0)) == "formato"        # a resposta é um pedaço do gabarito
    assert A.stage_of(cell(f=0.0, s=0.5)) == "busca"                  # faltou parágrafo de suporte
    assert A.stage_of(cell(f=0.0, s=1.0)) == "raciocinio"             # tinha toda a evidência e errou
    assert A.stage_of(cell(f=0.0, s=None, i=True)) == "raciocinio"    # sem suporte anotado: vale o texto do gabarito
    assert A.stage_of(cell(f=0.0, s=None, i=False)) == "busca"
    assert A.stage_of(cell(f=0.0, s=None, i=None)) == "indefinido"
    assert A.stage_of(cell(f=0.0, s=1.0, i=False)) == "raciocinio"    # com suporte anotado, o recall manda
    assert A.stage_of(cell(f=0.6, e=0.0, s=0.0), "em") == "busca"     # o critério muda quem é falha, não a etapa


def test_near_and_abstention():
    assert A.is_near("Obama", ["Barack Obama"]) and A.is_near("New York City", ["New York City, New York"])
    assert not A.is_near("Barack Obama", ["Barack Obama"])            # igual não é "pedaço" (isso é acerto)
    assert not A.is_near("Paris", ["Barack Obama"]) and not A.is_near("", ["x y"]) and not A.is_near("yes", ["yes"])
    assert not A.is_near("Obama Barack", ["Barack Hussein Obama"])    # tem de ser um trecho contíguo
    for p in ("", "  ", "Unknown", "The documents do not mention this.", "Cannot be determined", "No information available",
              "I don't know", "insufficient evidence", "Current occupation information not provided", "No evidence", "N/A"):
        assert A.is_abstention(p), p
    for p in ("No", "Graham Hill", "Nobody Knows (film)", "Unknown Pleasures", "The Unknown Soldier", "None of the Above (album)"):
        assert not A.is_abstention(p), p                                # "unknown" só conta quando é a resposta inteira


def test_answer_in_docs():
    docs = [{"id": "d1", "title": "Graham Hill", "text": "Norman Graham Hill was a British racing driver."}]
    assert A.answer_in_docs(["Graham Hill"], docs) is True
    assert A.answer_in_docs(["Jim Clark"], docs) is False
    assert A.answer_in_docs(["yes"], docs) is None and A.answer_in_docs(["no"], []) is None     # sim/não: não é localizável
    assert A.answer_in_docs(["Jim Clark"], []) is False
    assert A.answer_in_docs(["the British racing driver"], docs) is True                          # artigos e caixa não contam


def test_titles_and_docs_fallbacks():
    std = {"doc_titles": ["A", "B"], "contents": ['"A"\ntext a', '"B"\ntext b']}
    assert A.row_titles({"doc_titles": ["X"]}) == ["X"]
    assert A.row_titles({"contents": ['"T1"\nabc', "sem título"]}) == ["T1"]
    assert A.row_titles({"changed": False}, std) == ["A", "B"]           # resultado antigo, sem busca nova: os do Standard RAG
    assert A.row_titles({"changed": True}, std) is None                   # buscou de novo e não registrou: não dá para saber
    assert A.row_titles({"meta_action": "re-retrieve"}, std) is None
    assert A.row_titles({}) is None
    docs = A.row_docs({"changed": False}, None, std)
    assert [d["title"] for d in docs] == ["A", "B"] and docs[0]["text"] == "text a"
    assert A.title_of_contents('"Hustle & Flow"\nx') == "Hustle & Flow" and A.title_of_contents("no quotes") == ""


# ============================================================================ contagens
def _matrix(cols: dict, types=None, bases=None) -> dict:
    n = len(next(iter(cols.values())))
    return {"dataset": "t", "name": "T", "questions": [{"id": f"q{i}", "q": f"pergunta {i}", "gold": ["g"], "sup": ["S"],
                                                         "type": (types or ["a"] * n)[i], "hops": None} for i in range(n)],
            "methods": [{**A.method_info(k), "base": (bases or {}).get(k), "n": n, "n_errors": 0} for k in cols],
            "cells": cols, "has_support": True}


def test_compute_counts_overlap_breaks_and_stage_agreement():
    ok, busca, rac, fmt = cell(f=1.0, s=1.0), cell(f=0.0, s=0.0), cell(f=0.0, s=1.0), cell(f=0.2, a=1.0, s=1.0)
    cols = {"p/base": [ok, ok, busca, busca, ok, busca, None],
            "p/a": [ok, busca, busca, rac, ok, fmt, ok],
            "p/b": [ok, ok, busca, busca, rac, busca, ok]}
    m = _matrix(cols, types=["x", "x", "x", "y", "y", "y", "y"], bases={"p/a": "p/base", "p/b": "p/base"})
    st = A.compute(m, ["p/a", "p/b"])
    assert st["n"] == 7 and st["common"] == list(range(7))
    a, b = st["methods"]["p/a"], st["methods"]["p/b"]
    assert (a["fails"], a["stages"]) == (4, {"busca": 2, "raciocinio": 1, "formato": 1, "indefinido": 0})
    assert (b["fails"], b["stages"]["busca"], b["stages"]["raciocinio"]) == (4, 3, 1)
    assert (a["breaks"], a["fixes"], a["n_base"]) == (1, 0, 6)         # q1: a base acerta e A falha; a base não respondeu q6
    assert (b["breaks"], b["fixes"]) == (1, 0)                          # q4
    assert st["all_fail"]["idx"] == [2, 3, 5] and st["all_fail"]["n"] == 3
    assert set(st["all_fail"]) == {"n", "idx"}                          # só contagens, sem estatística inferencial
    assert st["hist"] == [2, 2, 3] and st["none_fail"] == 2
    assert st["same_stage"] == {"busca": 1, "misto": 2}                 # q2 busca/busca; q3 raciocínio×busca; q5 formato×busca
    p = st["pairs"][0]
    assert (p["both"], p["fa"], p["fb"]) == (3, 4, 4) and p["jaccard"] == pytest.approx(3 / 5)
    assert set(p) == {"a", "b", "fa", "fb", "both", "jaccard"}
    assert st["by_type"] == {"x": {"n": 3, "p/a": 2, "p/b": 1}, "y": {"n": 4, "p/a": 2, "p/b": 3}}
    # só entram os testes que TODOS os métodos escolhidos responderam
    assert A.compute(m, ["p/base", "p/a"])["n"] == 6
    assert A.compute(m, ["p/a", "nao/existe"])["keys"] == ["p/a"]
    # o critério muda a conta
    assert A.compute(m, ["p/a"], "f1", 0.1)["methods"]["p/a"]["fails"] == 3


def test_method_registry():
    i = A.method_info("ragcritic/ragcritic")
    assert i["family"] == "principal" and i["base"] == "ragcritic/standard_rag"
    assert A.method_info("ragcritic/ragcritic_rule_planning")["family"] == "ablacao"
    for gone in ("ragcritic_oracle_critic", "ragcritic_indent16", "ragcritic_official_feedback"):    # não estão no artigo
        assert A.method_info("ragcritic/" + gone)["family"] == "outro"
    assert A.method_info("dynakrag/dynakrag_cap2")["family"] == "variante"
    assert A.method_info("dynakrag/fixed_k12")["label"] == "Fixed-K (K=12)" and A.method_info("dynakrag/fixed_k12")["family"] == "base"
    assert A.method_info("dynakrag/dynakrag")["base"] == A.BEST_FIXED_K
    assert A.method_info("ragcritic/self_refine")["base"] == "ragcritic/standard_rag"
    assert A.method_info("outro/desconhecido")["family"] == "outro"
    assert set(A.PAPER_ORDER) == set(A.PAPER_LABEL) == {"ragcritic", "dynakrag"}
    keys = ["dynakrag/ircot", "dynakrag/dynakrag", "ragcritic/standard_rag", "ragcritic/flare", "ragcritic/ragcritic",
            "dynakrag/fixed_k12", "dynakrag/fixed_k3", "dynakrag/dynakrag_no_compression", "ragcritic/ragcritic_n2"]
    assert sorted(keys, key=A._order) == ["ragcritic/standard_rag", "ragcritic/ragcritic", "ragcritic/ragcritic_n2", "ragcritic/flare",
                                          "dynakrag/fixed_k3", "dynakrag/fixed_k12", "dynakrag/dynakrag",
                                          "dynakrag/dynakrag_no_compression", "dynakrag/ircot"]


# ============================================================================ com resultados de verdade (perfil smoke)
def test_matrix_from_smoke_run(smoke_cfg):
    m = A.load_matrix(smoke_cfg, "toy_multihop")
    keys = [x["key"] for x in m["methods"]]
    assert {"ragcritic/standard_rag", "ragcritic/ragcritic", "dynakrag/dynakrag", "dynakrag/fixed_k3"} <= set(keys)
    assert {k.split("/")[0] for k in keys} == {"ragcritic", "dynakrag"}
    assert not any("__" in k for k in keys)                              # arquivos de estágio não são métodos
    assert len(m["questions"]) == 12 and m["has_data"] and m["has_corpus"] and m["has_support"]
    info = {x["key"]: x for x in m["methods"]}
    assert info["dynakrag/dynakrag"]["base"].startswith("dynakrag/fixed_k")      # resolvido para o melhor Fixed-K
    assert info["ragcritic/ragcritic"]["base"] == "ragcritic/standard_rag"
    for k in keys:
        col = m["cells"][k]
        assert len(col) == 12 and all(c is not None for c in col)
        assert all(c["s"] is not None and 0.0 <= c["s"] <= 1.0 for c in col), k   # todo método registra o que recuperou
    assert A.default_methods(smoke_cfg, set(keys)) == ["ragcritic/standard_rag", "ragcritic/ragcritic", "dynakrag/dynakrag"]
    # um método que ainda não rodou não entra na seleção padrão
    assert A.default_methods(smoke_cfg, set(keys) - {"dynakrag/dynakrag"}) == ["ragcritic/standard_rag", "ragcritic/ragcritic"]
    sel = A.main_methods(m, A.default_methods(smoke_cfg, set(keys)))
    assert sel == ["ragcritic/ragcritic", "dynakrag/dynakrag"]
    st = A.compute(m, sel)
    assert st["n"] == 12 and sum(st["hist"]) == 12 and st["all_fail"]["n"] == st["hist"][-1]
    for k in sel:
        s = st["methods"][k]
        assert s["fails"] == sum(s["stages"].values()) and s["n"] == 12


def test_reports_and_csvs(smoke_cfg):
    out = A.write_reports(smoke_cfg)
    md = open(out["md"], encoding="utf-8").read()
    assert "## Toy multi-hop" in md and "Falhas em comum — RAG-Critic, DynaKRAG" in md and "Doctor-RAG" not in md
    assert "EM = 0" in md                                                # o critério de falha padrão
    with open(out["csv_testes"], encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    assert rows and {"dataset", "id", "metodo", "resposta", "f1", "falhou", "etapa", "quebra", "conserto"} <= set(rows[0])
    assert {r["etapa"] for r in rows} <= {"ok", "busca", "raciocinio", "formato", "indefinido"}
    assert all((r["falhou"] == "não") == (r["etapa"] == "ok") for r in rows)
    with open(out["csv_resumo"], encoding="utf-8-sig") as f:
        res = {(r["dataset"], r["artigo"], r["metodo"]): r for r in csv.DictReader(f, delimiter=";")}
    r = res[("toy_multihop", "ragcritic", "ragcritic")]
    assert int(r["falhas"]) == sum(int(r[s]) for s in A.STAGES) and "usa_gabarito" not in r and r["base"] == "ragcritic/standard_rag"
    assert {k[1] for k in res} == {"ragcritic", "dynakrag"}
    assert os.path.exists(out["csv_comum"])


def test_results_of_a_removed_paper_are_ignored(tmp_path):
    """Uma pasta de resultados de um artigo que saiu do projeto (ex.: ``doctorrag/``) não vira método nem relatório."""
    cfg = load_config("smoke", [], root=str(tmp_path))
    res = cfg["paths"]["results"]
    for paper, name in (("ragcritic", "standard_rag"), ("doctorrag", "react"), ("doctorrag", "react-drrag")):
        os.makedirs(os.path.join(res, paper, "toy_multihop"), exist_ok=True)
        with open(os.path.join(res, paper, "toy_multihop", name + ".jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"id": "q1", "question": "who?", "answers": ["Ann"], "pred": "Bob", "em": 0.0, "f1": 0.0,
                                "acc": 0.0, "doc_titles": ["A"]}) + "\n")
    assert A.result_files(cfg) == {"toy_multihop": {"ragcritic/standard_rag": os.path.join(res, "ragcritic", "toy_multihop",
                                                                                            "standard_rag.jsonl")}}
    assert [x["key"] for x in A.load_matrix(cfg, "toy_multihop")["methods"]] == ["ragcritic/standard_rag"]
    md = open(A.write_reports(cfg)["md"], encoding="utf-8").read()
    assert "Standard RAG" in md and "doctorrag" not in md and "ReAct" not in md


def test_old_results_without_doc_fields(tmp_path):
    """Resultados de antes desta versão (RAG-Critic sem doc_titles): a etapa sai dos documentos do Standard RAG."""
    cfg = load_config("smoke", [], root=str(tmp_path))
    res, data = cfg["paths"]["results"], os.path.join(cfg["paths"]["data"], "toy_multihop")
    os.makedirs(os.path.join(res, "ragcritic", "toy_multihop"))
    os.makedirs(data)

    def write(path, rows):
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    write(os.path.join(data, "eval.jsonl"), [{"id": "q1", "question": "who?", "answers": ["Ann Lee"], "supporting_titles": ["A", "B"]},
                                             {"id": "q2", "question": "where?", "answers": ["Oslo"], "supporting_titles": ["C"]}])
    std = [{"id": "q1", "pred": "Bob", "em": 0.0, "f1": 0.0, "acc": 0.0, "doc_titles": ["A", "B"], "contents": ['"A"\na', '"B"\nb']},
           {"id": "q2", "pred": "Rome", "em": 0.0, "f1": 0.0, "acc": 0.0, "doc_titles": ["Z"], "contents": ['"Z"\nz']}]
    old = [{"id": "q1", "pred": "Bob", "em": 0.0, "f1": 0.0, "acc": 0.0, "changed": False, "trace": []},
           {"id": "q2", "pred": "Rome", "em": 0.0, "f1": 0.0, "acc": 0.0, "changed": False, "trace": []}]
    write(os.path.join(res, "ragcritic", "toy_multihop", "standard_rag.jsonl"), std)
    write(os.path.join(res, "ragcritic", "toy_multihop", "ragcritic.jsonl"), old)
    m = A.load_matrix(cfg, "toy_multihop")
    stages = [A.stage_of(c) for c in m["cells"]["ragcritic/ragcritic"]]
    assert stages == ["raciocinio", "busca"] and not m["has_corpus"]
    assert [c["s"] for c in m["cells"]["ragcritic/ragcritic"]] == [1.0, 0.0]


# ============================================================================ o painel faz a mesma conta
@pytest.mark.skipif(not shutil.which("node"), reason="precisa do Node.js para rodar o JavaScript do painel")
def test_js_matches_python(smoke_cfg, tmp_path):
    m = A.load_matrix(smoke_cfg, "toy_multihop")
    keys = [x["key"] for x in m["methods"]]
    cases = [(keys, "f1", 0.5), (keys[:3], "em", 0.5), (keys[1:5], "acc", 0.5), (keys, "f1", 0.85), (keys[:2], "f1", 0.1)]
    mpath, spath = str(tmp_path / "m.json"), str(tmp_path / "run.js")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(m, f)
    with open(spath, "w", encoding="utf-8") as f:
        f.write("const core = require(%s); const m = require(%s); const cases = %s;\n"
                "const nn = v => (typeof v === 'number' && !isFinite(v)) ? null : v;\n"
                "const out = cases.map(c => core.compute(m, c[0], c[1], c[2]));\n"
                "console.log(JSON.stringify(out, (k, v) => nn(v)));\n" % (json.dumps(APP_JS), json.dumps(mpath), json.dumps(cases)))
    js = json.loads(subprocess.run(["node", spath], capture_output=True, text=True, check=True).stdout)

    def same(a, b, path=""):
        if isinstance(a, float) or isinstance(b, float):
            if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
                assert (a is None or math.isnan(a)) and (b is None or math.isnan(b)), path
            else:
                assert a == pytest.approx(b, rel=1e-9, abs=1e-12), path
        elif isinstance(a, dict):
            assert set(a) == set(b), path
            for k in a:
                same(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{path}[{i}]")
        else:
            assert a == b, path

    assert len(js) == len(cases)
    for (ks, metric, thr), got in zip(cases, js):
        same(A.compute(m, ks, metric, thr), got, f"compute({metric},{thr})")


def test_tabelas_no_padrao_dos_artigos(smoke_cfg):
    """Uma tabela por artigo, cada uma só com as métricas do próprio artigo (nada de misturar)."""
    import os

    from ragbench.tables import _rank, build_tables, write_tables

    assert _rank([0.5, 0.4, 0.3], True) == ["b", "u", ""] and _rank([10.0, 5.0, None], False) == ["", "b", ""]
    assert _rank([0.5, 0.5, 0.3], True) == ["b", "b", ""]                 # empate no topo: sem "segundo melhor"
    tabs = {t["id"]: t for t in build_tables(smoke_cfg)}
    assert set(tabs) == {"dynakrag", "ragcritic"}
    dk, rc = tabs["dynakrag"], tabs["ragcritic"]
    # DynaKRAG: EM, F1, Tok. — e só métodos do DynaKRAG
    assert set(dk["sub"]) == {"", "EM↑", "F1↑", "Tok.↓"}
    assert [r[1][0] for r in dk["rows"]][0] == "Fixed-K" and [r[1][0] for r in dk["rows"]][-1] == "DynaKRAG"
    assert not any(x in [r[1][0] for r in dk["rows"]] for x in ("RAG-Critic", "Standard RAG"))
    # RAG-Critic: EM e F1 por dataset + Overall F1 (sem tokens)
    assert "Tok.↓" not in rc["sub"] and rc["groups"][-1] == ("Overall", 1)
    assert [r[1][0] for r in rc["rows"]][0] == "Standard RAG" and [r[1][0] for r in rc["rows"]][-1] == "RAG-Critic"
    assert "(" in rc["rows"][-1][-1][0] and "(" not in rc["rows"][0][-1][0]         # ganho sobre o Standard RAG
    assert not any(x in [r[1][0] for r in rc["rows"]] for x in ("DynaKRAG", "Fixed-K"))
    for t in tabs.values():
        assert all(len(r) == len(t["sub"]) for r in t["rows"]) and sum(t["hl"]) >= 1
    path = write_tables(smoke_cfg)
    res = os.path.dirname(path)
    text = open(path, encoding="utf-8").read()
    assert "Uma tabela por artigo" in text and "figs/tabela_dynakrag.png" in text and "figs/tabela_ragcritic.png" in text
    assert "Doctor-RAG" not in text and "Repair" not in text
    for name in ("dynakrag", "ragcritic"):
        assert os.path.getsize(os.path.join(res, "figs", f"tabela_{name}.png")) > 5000
    tex = open(os.path.join(res, "tabelas.tex"), encoding="utf-8").read()
    assert "\\toprule" in tex and "\\textbf{" in tex and "$\\uparrow$" in tex and "\\usepackage{booktabs}" in tex
