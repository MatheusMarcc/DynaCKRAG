"""Testes rápidos (sem GPU): python -m pytest -q"""
import os
import subprocess
import sys

import pytest

from dynacritic.metrics import best_em_f1, f1_score, support_recall
from dynacritic.repair import validate_code
from dynacritic.state import EvidenceState
from dynacritic.validity import valid_actions

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_metrics():
    assert best_em_f1("The Eiffel Tower", ["eiffel tower"]) == (1.0, 1.0)
    assert f1_score("yes", "no") == 0.0
    assert 0 < f1_score("Barack Obama", "Obama") < 1
    assert support_recall(["A", "b "], ["a", "C"]) == 0.5


def test_initial_state_only_allows_retrieve_or_stop():
    s = EvidenceState(qid="q", question="Who?", ranking=[[0, 1.0], [1, 0.5]])
    assert set(valid_actions(s)) == {"retrieve_more", "stop_answer"}


def test_step_budget_forces_stop():
    s = EvidenceState(qid="q", question="Who?", ranking=[[0, 1.0]], history=["retrieve_more"] * 4)
    assert valid_actions(s) == ["stop_answer"]


@pytest.mark.parametrize("code", [
    "import os",
    "__import__('os').system('echo x')",
    "x = open('f.txt').read()",
    "x = doc_list.__class__",
    "while True:\n    pass",
    "f = lambda: 1",
    "final_answer = eval('1+1')",
])
def test_sandbox_rejects_unsafe_code(code):
    with pytest.raises(Exception):
        validate_code(code)


def test_sandbox_accepts_paper_style_plan():
    validate_code(
        'q = RewriteQuery(question, "clarify")[0]\n'
        "docs = Retrieval(q, 5)\n"
        'docs = [RefineDoc(q, d, "summarize") for d in docs]\n'
        'final_answer = GenerateAnswer(question, docs, "be precise")'
    )


def test_smoke_pipeline(tmp_path):
    """Pipeline inteiro com dados sintéticos e LLM mock."""
    import shutil

    shutil.copy(os.path.join(ROOT, "config.yaml"), tmp_path / "config.yaml")
    env = dict(os.environ, PYTHONPATH=ROOT)
    r = subprocess.run([sys.executable, "-m", "dynacritic", "smoke"], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    assert "smoke OK" in r.stdout
