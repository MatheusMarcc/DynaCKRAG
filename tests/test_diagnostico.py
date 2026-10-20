"""Diagnóstico das falhas de busca e raciocínio, e as métricas BLEU e ROUGE-L."""
import os

import pytest

from ragbench import cli
from ragbench.core.text import all_scores, bleu1_score, bleu_score, rougel_score
from ragbench.diagnostico import LABELS, gold_docs, is_ok, mixed_docs


def test_bleu_rouge_basics():
    assert bleu_score("Henry VIII", "Henry VIII") == pytest.approx(1.0)
    assert bleu_score("Andrew Davis", "Alfred Hitchcock") == 0.0
    assert bleu_score("", "x") == 0.0
    # valores do sentence_bleu do sacreBLEU (add-k, k = 1, ordem efetiva) sobre o texto normalizado
    assert bleu_score("King Henry VIII", "Henry VIII") == pytest.approx(0.6866, abs=1e-4)
    assert bleu_score("50 stars", "50") == pytest.approx(0.7071, abs=1e-4)
    assert bleu_score("the first Sunday in November", "the first Sunday in April") == pytest.approx(0.6580, abs=1e-4)
    # predição curta: precisão 1, penalidade de brevidade exp(1 - 5/3)
    assert bleu1_score("New York City", "New York City, New York") == pytest.approx(0.5134, abs=1e-4)
    assert rougel_score("King Henry VIII", "Henry VIII") == pytest.approx(0.8)
    assert rougel_score("x y z", "z y x") == pytest.approx(1 / 3)
    s = all_scores("50 stars", ["50", "fifty"])
    assert s["em"] == 0.0 and s["acc"] == 1.0 and s["f1"] == pytest.approx(0.6667, abs=1e-4)


def test_is_ok():
    assert is_ok({"acc": 1.0}) and not is_ok({"acc": 0.0})
    assert is_ok({"f1": 0.6}, "f1", 0.5) and not is_ok({"f1": 0.4}, "f1", 0.5)
    assert len(LABELS) == 4


def _corpus(docs):
    from ragbench.core.text import norm_title
    by_title = {}
    for d in docs:
        by_title.setdefault(norm_title(d["title"]), []).append(d)
    return {"by_id": {d["id"]: d for d in docs}, "by_title": by_title}


def test_gold_docs_picks_paragraph_with_answer():
    docs = [{"id": "d0", "title": "Kapalkundala", "text": "Kapalkundala is a novel by Bankim Chandra."},
            {"id": "d1", "title": "Bankim Chandra", "text": "Bankim was a writer from Bengal."},
            {"id": "d2", "title": "Bankim Chandra", "text": "His brother was Sanjib Chandra, also a writer."},
            {"id": "d3", "title": "Other", "text": "Unrelated."}]
    rec = {"question": "Who is the sibling of the author of Kapalkundala?", "answers": ["Sanjib Chandra"],
           "supporting_titles": ["Kapalkundala", "Bankim Chandra"]}
    got = [d["id"] for d in gold_docs(rec, _corpus(docs))]
    assert got == ["d0", "d2"]
    assert gold_docs({**rec, "supporting_titles": ["Missing"]}, _corpus(docs)) == []


def test_mixed_docs_keeps_size_and_inserts_missing():
    ret = [{"id": f"r{i}", "title": f"R{i}"} for i in range(5)]
    gold = [{"id": "g0", "title": "G0"}, {"id": "r1", "title": "R1"}]
    out = mixed_docs(ret, gold, "q1")
    ids = [d["id"] for d in out]
    assert len(out) == 5 and "g0" in ids and "r1" in ids and "r4" not in ids
    assert mixed_docs(ret, gold, "q1") == out                      # semente fixa por pergunta
    assert mixed_docs(ret, [{"id": "r2", "title": "R2"}], "q1") == ret   # nada faltando: igual à rodada


def test_cli_diagnostico_on_smoke(tmp_path, capsys):
    root = str(tmp_path)
    cli.main(["smoke", "--root", root], _exit=False)
    capsys.readouterr()
    cli.main(["diagnostico", "--profile", "smoke", "--root", root], _exit=False)
    out = capsys.readouterr().out
    assert "pergunta(s) testada(s)" in out
    res = os.path.join(root, "ragbench_results", "smoke", "diagnostico")
    md = open(os.path.join(res, "DIAGNOSTICO.md"), encoding="utf-8").read()
    assert "## RAG-Critic" in md and "## DynaKRAG" in md and "BLEU" in md and "ROUGE-L" in md
    assert "nosso" not in md.lower() and "p-valor" not in md
    assert os.path.exists(os.path.join(res, "diagnostico.csv"))
    cli.main(["diagnostico", "--profile", "smoke", "--root", root, "--metodo", "dynakrag",
              "--dataset", "toy_multihop", "--id", "toy_multihop-2"], _exit=False)
    out = capsys.readouterr().out
    assert "[suporte]" in out and "[suporte+busca]" in out and "Rótulo:" in out
