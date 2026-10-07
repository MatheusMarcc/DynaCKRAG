"""Linha de comando do ``bench``: nomes que não existem param a rodada na hora; servidor fechado sai sem traceback."""
import json
import os
import socket
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragbench import cli  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _log(root: str) -> str:
    """O que a rodada escreveu no log da pasta (as linhas com horário não passam pelo capsys)."""
    d = os.path.join(root, "ragbench_results", "logs")
    out = ""
    for name in sorted(os.listdir(d)):
        with open(os.path.join(d, name), encoding="utf-8") as f:
            out += f.read()
    return out


def test_bench_stops_on_a_method_that_does_not_exist(smoke_root, capsys):
    base = ["bench", "--profile", "smoke", "--root", smoke_root]
    status_path = os.path.join(smoke_root, "ragbench_results", "smoke", "status.json")
    with open(status_path, encoding="utf-8") as f:
        before = f.read()
    # o caso que aconteceu: o terminal colou outro texto no fim do último nome
    glued = r"ragcritic/ragcritic_rule_planningc:\Users\x\DynaKRAGKritic\.venv\Scripts\activate"
    with pytest.raises(SystemExit) as e:
        cli.main(base + ["--methods", "dynakrag/dynakrag", "ragcritic/ragcritic_zeroshot_critic", glued], _exit=False)
    out = capsys.readouterr().out
    assert e.value.code == 2
    assert f"método desconhecido: {glued}" in out and "você quis dizer ragcritic/ragcritic_rule_planning?" in out
    assert "ragcritic/  errorbench, standard_rag, ragcritic" in out and "dynakrag/  fixed_k" in out
    assert "nada foi rodado" in out and "jobs na fila" not in out
    with open(status_path, encoding="utf-8") as f:
        assert f.read() == before
    # erro de digitação, com e sem o artigo na frente; o artigo errado também não passa
    for bad, guess in (("dynakrag/dinakrag", "dynakrag/dynakrag"), ("ragcritic_rule_planing", "ragcritic_rule_planning"),
                       ("dynakrag/ragcritic_rule_planning", None), ("dynakrag/train_controller", None),
                       ("ragcritic/ragcritic_oracle_critic", None), ("ragcritic_indent16", None)):       # nomes fora do catálogo
        with pytest.raises(SystemExit) as e:
            cli.main(base + ["--dry-run", "--methods", bad], _exit=False)
        out = capsys.readouterr().out
        assert e.value.code == 2 and f"método desconhecido: {bad}" in out
        if guess:
            assert f"você quis dizer {guess}?" in out


def test_bench_stops_on_a_dataset_that_does_not_exist(smoke_root, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["bench", "--profile", "smoke", "--root", smoke_root, "--dry-run", "--datasets", "toy_multihop", "hotpotqaa"],
                 _exit=False)
    out = capsys.readouterr().out
    assert e.value.code == 2 and "dataset desconhecido: hotpotqaa" in out and "hotpotqa, 2wiki" in out
    assert "métodos do perfil" not in out


def test_bench_accepts_the_names_that_exist(smoke_root, capsys):
    base = ["bench", "--profile", "smoke", "--root", smoke_root, "--dry-run"]
    cli.main(base + ["--datasets", "toy_multihop", "--methods", "dynakrag/dynakrag", "ragcritic_rule_planning",
                     "ragcritic/ragcritic_zeroshot_critic", "errorbench"], _exit=False)
    out = capsys.readouterr().out
    for key in ("dynakrag/toy_multihop/train_controller", "dynakrag/toy_multihop/dynakrag",
                "ragcritic/toy_multihop/ragcritic_rule_planning", "ragcritic/toy_multihop/ragcritic_zeroshot_critic",
                "ragcritic/errorbench/rag-critic-3b"):
        assert key in out, key
    assert "fixed_k" not in out and "standard_rag" not in out
    # todo nome do catálogo é aceito, com o artigo na frente
    from ragbench.config import load_config

    catalog = cli.method_catalog(load_config("smoke", [], root=smoke_root))
    assert "ragcritic_rule_planning" in catalog["ragcritic"] and "dynakrag" in catalog["dynakrag"]
    assert not [n for n in catalog["ragcritic"] if "oracle" in n or "indent" in n or "official" in n]
    cli.main(base + ["--methods"] + [f"{p}/{n}" for p, names in catalog.items() for n in names], _exit=False)
    assert "desconhecido" not in capsys.readouterr().out


def test_selection_that_picks_nothing_is_said_out_loud(tmp_path, capsys):
    root = str(tmp_path)
    base = ["--profile", "smoke", "--root", root]
    cli.main(["prepare"] + base, _exit=False)
    assert "nenhuma etapa na fila" not in _log(root)
    cli.main(["bench"] + base + ["--paper", "dynakrag", "--methods", "ragcritic/standard_rag", "--no-report"], _exit=False)
    assert "nenhuma etapa na fila" in _log(root)


def test_bench_with_the_server_closed_exits_without_traceback(tmp_path, monkeypatch, capsys):
    from ragbench.core import server as S

    monkeypatch.setattr(S, "wait_up", lambda *a, **k: False)
    monkeypatch.setattr("time.sleep", lambda s: pytest.fail("não devia esperar para tentar de novo"))
    root = str(tmp_path)
    base = ["--profile", "smoke", "--root", root]
    cli.main(["prepare"] + base, _exit=False)
    dead = ["--set", "llm.backend=ollama", "--set", f"llm.base_url=http://127.0.0.1:{_free_port()}"]
    capsys.readouterr()
    with pytest.raises(SystemExit) as e:
        cli.main(["bench"] + base + dead + ["--paper", "dynakrag", "--tier", "1"])
    cap = capsys.readouterr()
    assert e.value.code == 1
    log = _log(root)
    assert "o Ollama não está respondendo" in log and "Rode de novo o mesmo comando" in log
    assert "tentando de novo em" not in log
    assert "Traceback" not in cap.out + cap.err + log
    assert cap.out.rstrip().splitlines()[-2].startswith("parou: o Ollama não está respondendo")     # o motivo fica no fim
    assert "rode o mesmo comando de novo" in cap.out.rstrip().splitlines()[-1]
    res = os.path.join(root, "ragbench_results", "smoke")
    with open(os.path.join(res, "status.json"), encoding="utf-8") as f:
        status = json.load(f)
    assert [v["state"] for v in status.values()] == ["failed"]            # parou na primeira etapa
    assert "Abra o app do Ollama" in next(iter(status.values()))["error"]
    assert os.path.exists(os.path.join(res, "REPORT.md"))                 # os relatórios saem mesmo assim
    # um comando avulso também termina com a mensagem, sem traceback
    with pytest.raises(SystemExit) as e:
        cli.main(["dynakrag", "run", "--dataset", "toy_multihop", "--method", "fixed_k3", "--limit", "2"] + base + dead)
    cap = capsys.readouterr()
    assert e.value.code == 1 and "parou: o Ollama não está respondendo" in cap.out and "Traceback" not in cap.out + cap.err
