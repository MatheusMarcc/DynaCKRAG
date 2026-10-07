"""Testes do RAG-Critic: fidelidade dos prompts D/E, parser do crítico, sandbox do executor, métricas oficiais."""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragbench.core.retrieval import BM25Index, Retriever  # noqa: E402
from ragbench.ragcritic import prompts as P  # noqa: E402
from ragbench.ragcritic.critic import parse_critic_output  # noqa: E402
from ragbench.ragcritic.errorbench import official_metrics, sample, swap_indent  # noqa: E402
from ragbench.ragcritic.executor import Executor, ProgramError, final_answer_var, validate  # noqa: E402
from ragbench.ragcritic.planner import extract_code  # noqa: E402
from ragbench.ragcritic.taxonomy import RULE_PROGRAMS, canonical_tier1, mapping_table_text, rule_program  # noqa: E402

BENCH = os.environ.get("RAGBENCH_ERRORBENCH_DIR")  # pasta com critic_model_test.jsonl e baseline_test.jsonl (opcional)


def test_render_parse_roundtrip():
    ins = P.render_critic("who?", [f'"T{i}"\ntext {i}' for i in range(5)], "So the final answer is: X", 20, "critic")
    p = P.parse_critic_instruction(ins)
    assert p["question"] == "who?" and len(p["passages"]) == 5 and p["indent"] == 20 and p["kind"] == "critic"
    assert P.render_critic(p["question"], p["passages"], p["prediction"], 20, "critic") == ins
    sw = swap_indent(ins)
    assert P.parse_critic_instruction(sw)["indent"] == 16 and sw != ins


@pytest.mark.skipif(not BENCH, reason="defina RAGBENCH_ERRORBENCH_DIR para testar com o benchmark oficial")
def test_prompts_match_official_bench_byte_by_byte():
    for fn, kind in (("critic_model_test.jsonl", "critic"), ("baseline_test.jsonl", "baseline")):
        with open(os.path.join(BENCH, fn), encoding="utf-8") as f:
            for line in f:
                ins = json.loads(line)["instruction"]
                p = P.parse_critic_instruction(ins)
                assert p["kind"] == kind
                assert P.render_critic(p["question"], p["passages"], p["prediction"], p["indent"], kind) == ins


def test_parse_critic_output():
    o = parse_critic_output("{'Judgement': 'Error', 'Error_analysis': 'x', 'tag1': ['Incomplete Information'], 'tag2': [], 'tag3': []}")
    assert o["judgement"] == "Error" and o["tag1"] == ["Incomplete Information"] and o["parse_ok"]
    assert parse_critic_output('{"Judgement": "Correct"}')["judgement"] == "Correct"
    assert parse_critic_output("Judgement: Incorrect, the answer...")["judgement"] == "Error"
    assert parse_critic_output("garbage")["judgement"] == "Unknown"


def test_taxonomy():
    assert canonical_tier1("Incomplete Response") == "Incomplete or Missing Response"
    assert "Re-retrieval" in mapping_table_text(["Off-Topic Response"])
    assert rule_program(["Inaccurate Response"])[0] == "Inaccurate or Misunderstood Response"
    for code in RULE_PROGRAMS.values():
        validate(code)  # todos os programas da tabela T passam na sandbox


def test_extract_code_and_answer_var():
    out = "Here is the code:\n```python\nx = Retrieval(query=question, topk=3)\nfinal_answer = GenerateAnswer(query=question, docs=x)\n```"
    code = extract_code(out)
    assert code.startswith("x = Retrieval") and final_answer_var(code) == "final_answer"


@pytest.mark.parametrize("bad", [
    "import os", "__import__('os')", "open('x')", "x = (1).__class__", "while True:\n    pass",
    "def f():\n    pass", "eval('1')", "lambda: 1", "'{0.__class__}'.format(1)", "x = [1] * 10**9"])
def test_sandbox_rejects(bad):
    with pytest.raises(ProgramError):
        validate(bad)


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, model, messages, task, **kw):
        self.calls.append(task)
        if task == "rc_rewrite_clarify":
            return '{"query": "clarified"}'
        if task == "rc_decompose":
            return '["a", "b"]'
        if task.startswith("rc_refine"):
            return '{"refined_document": "refined"}'
        return "So the final answer is: Paris"


def _executor(max_calls=12):
    docs = [{"id": f"d{i}", "title": f"T{i}", "text": f"text {i} capital"} for i in range(8)]
    return Executor(_FakeLLM(), Retriever(docs, BM25Index().build(docs)), max_calls=max_calls)


def test_executor_runs_case_study_program():
    code = ('clarified_query = RewriteQuery(query=question, instruction="clarify")\n'
            'retrieved_documents = Retrieval(query=clarified_query[0], topk=5)\n'
            'summarized = [RefineDoc(query=question, doc=doc, instruction="summarize") for doc in retrieved_documents]\n'
            'final_answer = GenerateAnswer(query=question, docs=summarized, additional_instruction="be precise")')
    info = _executor().run(code, "capital?", ["doc a", "doc b"], "Rome")
    assert info["ok"] and "Paris" in info["answer_raw"] and info["n_calls"] == 8


def test_executor_coercions_and_loop_guard():
    code = ("docs = doc_list\n"
            "for d in docs:\n"
            "    docs.append(d)\n"               # lista que cresce: o guard materializa e limita
            "final_answer = GenerateAnswer(question, docs)")
    info = _executor().run(code, "q", ["a", "b"], "p")
    assert info["ok"]
    info = _executor().run("final_answer = GenerateAnswer(query=['q1','q2'], docs=[['x'], 'y'])", "q", [], "p")
    assert info["ok"]


def test_executor_budget_and_errors():
    code = "\n".join(f"x{i} = RewriteQuery(query=question, instruction='clarify')" for i in range(5)) + \
           "\nfinal_answer = GenerateAnswer(query=question, docs=doc_list)"
    info = _executor(max_calls=3).run(code, "q", ["a"], "p")
    assert not info["ok"] and "orçamento" in info["error"]
    info = _executor().run("x = Retrieval(query=question, topk=3)", "q", ["a"], "p")
    assert not info["ok"]  # sem GenerateAnswer


def test_official_metrics_semantics():
    golds = [{"Judgement": "Correct"}, {"Judgement": "Error", "tag1": ["Incomplete Information"], "tag2": ["Lack of Specificity"]},
             {"Judgement": "Error", "tag1": ["Irrelevant Information"], "tag2": ["Irrelevant Information Retrieval"]}]
    outs = [{"Judgement": "Correct"}, {"Judgement": "Error", "tag1": ["Incomplete Information"], "tag2": []},
            {"Judgement": "Correct", "tag1": [], "tag2": []}]
    m = official_metrics(golds, outs)
    assert m["identification"]["correct"] == 1.0 and m["identification"]["error"] == 0.5
    assert abs(m["identification"]["avg"] - 2 / 3) < 1e-9
    assert m["classification"]["tag1_acc"] == 0.5  # 1 acerto em 2 tags gold contabilizadas


def test_sample_stratified():
    items = [{"id": str(i), "label": "Error" if i % 2 else "Correct", "gold": {"tag2": [f"t{i % 5}"]}} for i in range(100)]
    s = sample(items, 10, 10, 13)
    assert sum(x["label"] == "Correct" for x in s) == 10 and sum(x["label"] == "Error" for x in s) == 10
