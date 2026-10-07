"""Leitura e escrita com o painel aberto: no Windows, um arquivo em uso por outro processo dá PermissionError."""
import os

import pytest

from ragbench.core import io


def _flaky(real, fails):
    state = {"n": 0}

    def fn(*a, **k):
        state["n"] += 1
        if state["n"] <= fails:
            raise PermissionError(13, "arquivo em uso por outro processo")
        return real(*a, **k)
    return fn, state


def test_troca_de_arquivo_em_uso_e_repetida_no_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "_WINDOWS", True)
    monkeypatch.setattr(io, "_WAIT", 0)
    fn, state = _flaky(os.replace, 2)
    monkeypatch.setattr(io.os, "replace", fn)
    p = str(tmp_path / "status.json")
    io.write_json(p, {"job": "done"})
    assert state["n"] == 3
    assert io.read_json(p) == {"job": "done"}
    state["n"] = 0
    q = str(tmp_path / "linhas.jsonl")
    io.write_jsonl(q, [{"id": 1}, {"id": 2}])
    assert state["n"] == 3
    assert [r["id"] for r in io.read_jsonl(q)] == [1, 2]
    assert not os.path.exists(p + ".tmp") and not os.path.exists(q + ".tmp")


def test_leitura_de_arquivo_em_uso_e_repetida_no_windows(tmp_path, monkeypatch):
    p = str(tmp_path / "status.json")
    io.write_json(p, {"a": 1})
    monkeypatch.setattr(io, "_WINDOWS", True)
    monkeypatch.setattr(io, "_WAIT", 0)
    import builtins
    fn, state = _flaky(builtins.open, 3)
    monkeypatch.setattr(builtins, "open", fn)
    assert io.read_json(p) == {"a": 1}
    assert state["n"] == 4


def test_bloqueio_que_nao_passa_vira_erro(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "_WINDOWS", True)
    monkeypatch.setattr(io, "_WAIT", 0)
    monkeypatch.setattr(io, "_TRIES", 4)
    fn, state = _flaky(os.replace, 99)
    monkeypatch.setattr(io.os, "replace", fn)
    with pytest.raises(PermissionError):
        io.write_json(str(tmp_path / "x.json"), {})
    assert state["n"] == 4


def test_fora_do_windows_o_erro_sobe_na_hora(tmp_path, monkeypatch):
    monkeypatch.setattr(io, "_WINDOWS", False)
    fn, state = _flaky(os.replace, 1)
    monkeypatch.setattr(io.os, "replace", fn)
    with pytest.raises(PermissionError):
        io.write_json(str(tmp_path / "x.json"), {})
    assert state["n"] == 1


def test_move_file_retries_when_windows_says_in_use(tmp_path, monkeypatch):
    """Guardar um resultado antigo enquanto o painel o lê: no Windows o move falha por instantes e é repetido."""
    import shutil

    from ragbench.core import io

    src, dst = tmp_path / "a.jsonl", tmp_path / "guardado" / "a.jsonl"
    src.write_text("{}\n", encoding="utf-8")
    calls = {"n": 0}
    real = shutil.move

    def flaky(a, b):
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError("em uso")
        return real(a, b)

    monkeypatch.setattr(shutil, "move", flaky)
    monkeypatch.setattr(io, "_WINDOWS", True)
    monkeypatch.setattr(io, "_WAIT", 0.0)
    assert io.move_file(str(src), str(dst)) == str(dst)
    assert calls["n"] == 3 and dst.exists() and not src.exists()
