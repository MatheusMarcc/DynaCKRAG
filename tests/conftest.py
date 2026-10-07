"""Fixtures compartilhadas: uma rodada pequena (perfil smoke, LLM falso) feita UMA vez por sessão de testes."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="session")
def smoke_root(tmp_path_factory):
    """Pasta com os dados de teste preparados e o tier 1 dos dois artigos já rodado (sem GPU, sem internet)."""
    from ragbench import cli
    from ragbench.core import datasets as D
    from ragbench.core import retrieval as R

    root = str(tmp_path_factory.mktemp("ragbench_smoke"))
    D.clear_cache()
    R.clear_cache()
    base = ["--profile", "smoke", "--root", root]
    cli.main(["prepare"] + base, _exit=False)
    cli.main(["bench"] + base + ["--tier", "1"], _exit=False)
    return root


@pytest.fixture(scope="session")
def smoke_cfg(smoke_root):
    from ragbench.config import load_config

    return load_config("smoke", [], root=smoke_root)
