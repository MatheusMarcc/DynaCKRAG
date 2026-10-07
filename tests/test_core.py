"""Testes do núcleo: métricas, extração de resposta, JSON tolerante, BM25, conversores de datasets."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragbench.core.datasets import build_corpus, conv_hotpot_like, conv_musique, conv_tevatron  # noqa: E402
from ragbench.core.jsonutil import as_str_list, parse_obj  # noqa: E402
from ragbench.core.retrieval import BM25Index, HashEncoder, Retriever  # noqa: E402
from ragbench.core.stats import safe_mean, summarize  # noqa: E402
from ragbench.core.text import best_em_f1, extract_final_answer, f1_score, normalize_answer  # noqa: E402


def test_normalize_and_em_f1():
    assert normalize_answer("The Eiffel Tower!") == "eiffel tower"
    assert best_em_f1("the Eiffel tower", ["Eiffel Tower"]) == (1.0, 1.0)
    assert f1_score("yes", "no") == 0.0
    em, f1 = best_em_f1("Graham Hill of England", ["Graham Hill"])
    assert em == 0.0 and 0.6 < f1 < 0.9


def test_extract_final_answer():
    assert extract_final_answer("reasoning...\nSo the final answer is: Paris.") == "Paris"
    assert extract_final_answer("So the answer is: 1962") == "1962"
    assert extract_final_answer("Kestrel Bay\nextra") == "Kestrel Bay"
    assert extract_final_answer("Final answer: **Nikola Tesla**") == "Nikola Tesla"


def test_parse_obj_variants():
    assert parse_obj("{'Judgement': 'Correct'}") == {"Judgement": "Correct"}
    assert parse_obj('```json\n{"sufficient": false}\n```') == {"sufficient": False}
    assert parse_obj('blah {"a": 1} blah') == {"a": 1}
    assert as_str_list({"query": "x"}) == ["x"]
    assert as_str_list(["a", {"query": "b"}]) == ["a", "b"]


def test_bm25_and_retriever_exclusion():
    docs = [{"id": f"d{i}", "title": t, "text": x} for i, (t, x) in enumerate([
        ("Paris", "Paris is the capital of France."), ("Berlin", "Berlin is the capital of Germany."),
        ("Rome", "Rome is the capital of Italy."), ("France", "France is a country in Europe.")])]
    r = Retriever(docs, BM25Index().build(docs))
    top = r.search("capital of France", 2)
    assert top[0]["title"] == "Paris"
    top2 = r.search("capital of France", 2, exclude_ids={"d0"})
    assert all(h["id"] != "d0" for h in top2)


def test_hash_encoder_normalized():
    e = HashEncoder().encode(["hello world", "another text"])
    assert e.shape == (2, 256)
    assert abs((e[0] ** 2).sum() - 1) < 1e-5


def test_converters():
    hp = {"id": "x1", "question": "Q?", "answer": "A", "type": "bridge", "level": "hard",
          "context": {"title": ["T1", "T2"], "sentences": [["s1.", "s2."], ["s3."]]},
          "supporting_facts": {"title": ["T1", "T1"], "sent_id": [0, 1]}}
    r = conv_hotpot_like(hp, "hotpotqa")
    assert r["supporting_titles"] == ["T1"] and r["paragraphs"][0]["text"] == "s1. s2."
    wiki = {"_id": "w1", "question": "Q?", "answer": "A", "type": "compositional",
            "context": [["T1", ["a."]], ["T2", ["b."]]], "supporting_facts": [["T2", 0]]}
    r = conv_hotpot_like(wiki, "2wiki")
    assert r["supporting_titles"] == ["T2"] and r["type"] == "compositional"
    mu = {"id": "3hop1__1_2_3", "question": "Q?", "answer": "A", "answer_aliases": ["AA"], "answerable": True,
          "paragraphs": [{"idx": 0, "title": "P0", "paragraph_text": "t0", "is_supporting": True},
                         {"idx": 1, "title": "P1", "paragraph_text": "t1", "is_supporting": False}]}
    r = conv_musique(mu)
    assert r["hops"] == 3 and r["answers"] == ["A", "AA"] and r["supporting_titles"] == ["P0"]
    assert conv_musique({**mu, "answerable": False}) is None
    tv = {"query_id": "1", "query": "who?", "answers": ["x"], "positive_passages": [{"docid": "a", "title": "T", "text": "t"}],
          "negative_passages": [{"docid": "b", "title": "N", "text": "n"}]}
    r = conv_tevatron(tv, "nq")
    assert len(r["paragraphs"]) == 2 and r["supporting_titles"] == ["T"]


def test_build_corpus_dedup():
    recs = [{"paragraphs": [{"title": "A", "text": "x"}, {"title": "B", "text": "y"}]},
            {"paragraphs": [{"title": "A", "text": "x"}, {"title": "A", "text": "different"}]}]
    c = build_corpus(recs)
    assert len(c) == 3 and [d["id"] for d in c] == ["d0", "d1", "d2"]


def test_stats():
    assert safe_mean([1, None, float("nan"), 3]) == 2 and safe_mean([]) != safe_mean([])
    s = summarize([{"em": 1.0, "f1": 1.0, "tokens": 10}, {"em": 0.0, "f1": 0.5, "tokens": 30, "error": "x"}])
    assert (s["n"], s["em"], s["f1"], s["tokens"], s["n_errors"]) == (2, 0.5, 0.75, 20, 1)
    assert "f1_ci" not in s
