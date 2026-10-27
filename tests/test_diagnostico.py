"""Diagnóstico das falhas de busca e raciocínio, as métricas BLEU e Levenshtein e o baseline LLM base."""
import os

import pytest

from ragbench import cli
from ragbench.core.text import all_scores, bleu_score, lev_similarity, levenshtein
from ragbench.diagnostico import LABELS, gold_docs, is_ok, mixed_docs


def test_bleu_rouge_basics():
    assert bleu_score("Henry VIII", "Henry VIII") == pytest.approx(1.0)
    assert bleu_score("Andrew Davis", "Alfred Hitchcock") == 0.0
    assert bleu_score("", "x") == 0.0
    # valores do sentence_bleu do sacreBLEU (add-k, k = 1, ordem efetiva) sobre o texto normalizado
    assert bleu_score("King Henry VIII", "Henry VIII") == pytest.approx(0.6866, abs=1e-4)
    assert bleu_score("50 stars", "50") == pytest.approx(0.7071, abs=1e-4)
    assert bleu_score("the first Sunday in November", "the first Sunday in April") == pytest.approx(0.6580, abs=1e-4)
    # resposta curta: penalidade de brevidade exp(1 - 3/2) sobre a média geométrica das precisões
    assert bleu_score("Gemma Whelan", "Gemma Elizabeth Whelan") == pytest.approx(0.5100, abs=1e-4)
    s = all_scores("50 stars", ["50", "fifty"])
    assert set(s) == {"em", "f1", "bleu", "lev"}
    assert s["em"] == 0.0 and s["f1"] == pytest.approx(0.6667, abs=1e-4) and s["bleu"] == pytest.approx(0.7071, abs=1e-4)


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3 and levenshtein("", "abc") == 3 and levenshtein("abc", "abc") == 0
    assert lev_similarity("Henry VIII", "henry viii.") == 1.0            # mesma normalização do EM
    assert lev_similarity("King Henry VIII", "Henry VIII") == pytest.approx(1 - 5 / 15)
    assert lev_similarity("3", "three") == 0.0
    assert lev_similarity("", "") == 1.0 and lev_similarity("x", "") == 0.0
    assert all_scores("Gemma Whelan", ["Gemma Elizabeth Whelan"])["lev"] == pytest.approx(0.5455, abs=1e-4)


def test_is_ok():
    assert is_ok({"bleu": 0.5}) and not is_ok({"bleu": 0.49})
    assert is_ok({"em": 1.0}, "em") and not is_ok({"em": 0.0}, "em")
    assert is_ok({"f1": 0.6}, "f1", 0.5) and not is_ok({"f1": 0.4}, "f1", 0.5)
    assert is_ok({"lev": 0.7}, "lev") and not is_ok({"lev": 0.69}, "lev")   # corte padrão do Lev: 0,7
    assert is_ok({"lev": 0.5}, "lev", 0.5)
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
    assert "## RAG-Critic" in md and "## DynaKRAG" in md and "BLEU" in md and "Lev" in md and "ROUGE" not in md and "Acc" not in md
    assert "nosso" not in md.lower() and "p-valor" not in md
    assert os.path.exists(os.path.join(res, "diagnostico.csv"))
    cli.main(["diagnostico", "--profile", "smoke", "--root", root, "--metodo", "dynakrag",
              "--dataset", "toy_multihop", "--id", "toy_multihop-2"], _exit=False)
    out = capsys.readouterr().out
    assert "[suporte]" in out and "[suporte+busca]" in out and "Rótulo:" in out


def test_cli_llm_base_on_smoke(tmp_path, capsys):
    root = str(tmp_path)
    cli.main(["smoke", "--root", root], _exit=False)
    capsys.readouterr()
    cli.main(["llm-base", "--profile", "smoke", "--root", root], _exit=False)
    out = capsys.readouterr().out
    res = os.path.join(root, "ragbench_results", "smoke", "llm_base")
    assert "relatorio:" in out and os.path.exists(os.path.join(res, "toy_multihop.jsonl"))
    md = open(os.path.join(res, "LLM_BASE.md"), encoding="utf-8").read()
    assert "LLM base (sem recuperação)" in md and "DynaKRAG" in md and "RAG-Critic" in md and "Lev" in md
    assert "nosso" not in md.lower() and "p-valor" not in md
    from ragbench.core.io import read_jsonl
    rows = read_jsonl(os.path.join(res, "toy_multihop.jsonl"))
    assert rows and all(r["n_ret"] == 0 and r["doc_titles"] == [] for r in rows)
