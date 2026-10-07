"""Testa o caminho HTTP do backend Ollama (com um servidor falso): chat, JSON schema, ChatML cru, logprobs,
404 de modelo ausente, doctor e um mini benchmark ponta a ponta pelo HTTP."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fake_ollama  # noqa: E402
from ragbench.config import load_config  # noqa: E402
from ragbench.core.llm import LLM, LLMError  # noqa: E402
from ragbench.core.tokens import disable_tokenizer  # noqa: E402

disable_tokenizer()


@pytest.fixture(scope="module")
def server():
    srv, url = fake_ollama.start()
    yield url
    srv.shutdown()


def _cfg(tmp_path, url, extra=()):
    return load_config("smoke", ["llm.backend=ollama", f"llm.base_url={url}", "llm.retries=0", *extra], root=str(tmp_path))


def test_chat_schema_chatml_logprobs(tmp_path, server):
    cfg = _cfg(tmp_path, server)
    llm = LLM(cfg)
    out = llm.chat("generator", [{"role": "user", "content": "Evidence:\n[1] A: x\n\nQuestion: who?\nAnswer:"}], task="t")
    assert isinstance(out, str) and out
    j = llm.chat("generator", [{"role": "user", "content": "Question: q\n\nDocuments:\n[1] a"}], task="t2",
                 fmt={"type": "object", "properties": {"sufficient": {"type": "boolean"}}})
    assert "sufficient" in j
    r = llm.chat_full("critic", [{"role": "user", "content": "hello"}], task="t3", template="chatml")
    last = fake_ollama.LOG[-1]
    assert last["path"] == "/api/generate" and last["body"]["raw"] is True
    assert last["body"]["prompt"].startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.")
    assert last["body"]["prompt"].endswith("<|im_start|>assistant\n")
    assert "<|im_end|>" in last["body"]["options"]["stop"]
    r = llm.chat_full("generator", [{"role": "user", "content": "**Question**: q **Response**:"}], task="t4", logprobs=True)
    assert r["logprobs"] and "logprob" in r["logprobs"][0]
    # cache: a mesma chamada não vai ao servidor de novo
    n = len(fake_ollama.LOG)
    llm.chat("generator", [{"role": "user", "content": "Evidence:\n[1] A: x\n\nQuestion: who?\nAnswer:"}], task="t")
    assert len(fake_ollama.LOG) == n


def test_missing_model_raises(tmp_path, server):
    cfg = _cfg(tmp_path, server, ["models.generator=nao-existe:1b"])
    with pytest.raises(LLMError):
        LLM(cfg).chat("generator", [{"role": "user", "content": "x"}], task="t")


def test_doctor_and_mini_bench_over_http(tmp_path, server, capsys):
    from ragbench import cli

    root = str(tmp_path)
    base = ["--profile", "smoke", "--root", root, "--set", "llm.backend=ollama", "--set", f"llm.base_url={server}",
            "--set", "ragcritic.errorbench.zeroshot_models=[qwen2.5:3b]"]
    cli.main(["prepare"] + base, _exit=False)
    with pytest.raises(SystemExit) as e:
        cli.main(["doctor"] + base)
    out = capsys.readouterr().out
    assert "na GPU" in out and "logprobs (FLARE): suportado" in out
    assert e.value.code == 0
    cli.main(["bench"] + base + ["--tier", "1"], _exit=False)
    res = os.path.join(root, "ragbench_results", "smoke")
    assert os.path.exists(os.path.join(res, "ragcritic", "toy_single", "ragcritic.jsonl"))
    assert os.path.exists(os.path.join(res, "dynakrag", "toy_multihop", "dynakrag.jsonl"))
    import json

    status = json.load(open(os.path.join(res, "status.json"), encoding="utf-8"))
    assert all(v["state"] == "done" for v in status.values()), status


# ----------------------------------------------------------------------------- servidor fechado
MSG = [{"role": "user", "content": "Evidence:\n[1] A: x\n\nQuestion: who?\nAnswer:"}]
MSG2 = [{"role": "user", "content": "Evidence:\n[1] B: y\n\nQuestion: where?\nAnswer:"}]


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_closed_server_stops_at_once_and_says_what_to_do(tmp_path, monkeypatch):
    """Nada responde desde o começo da rodada: para na hora, sem as tentativas com espera, e diz o que fazer."""
    from ragbench.core import server as S

    naps, opened = [], []
    monkeypatch.setattr(S, "wait_up", lambda *a, **k: False)
    monkeypatch.setattr(S, "start_local_ollama", lambda *a, **k: opened.append(a) or "x")
    monkeypatch.setattr("time.sleep", naps.append)
    url = f"http://127.0.0.1:{_free_port()}"
    llm = LLM(load_config("smoke", ["llm.backend=ollama", f"llm.base_url={url}"], root=str(tmp_path)))
    assert llm.retries == 3
    with pytest.raises(LLMError) as e:
        llm.chat("generator", MSG, task="t")
    assert "não está respondendo" in str(e.value) and "ollama serve" in str(e.value) and url in str(e.value)
    assert not naps                       # nenhuma espera de 3/10/30/60 s
    assert not opened                     # não é o Ollama desta máquina (porta padrão): não tenta abrir nada
    with pytest.raises(LLMError):         # e as chamadas seguintes param na hora também
        llm.chat("generator", MSG2, task="t")
    assert not naps and not opened


def test_local_ollama_is_opened_and_the_call_goes_on(tmp_path, monkeypatch):
    """Ollama desta máquina fechado na primeira chamada: abre, repete a chamada (sem gastar tentativa) e segue."""
    from ragbench.core import server as S

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    opened = []

    def fake_start(base_url, wait_s=45.0):
        assert base_url == url
        opened.append(fake_ollama.start(port)[0])
        return "ollama serve"

    monkeypatch.setattr(S, "OLLAMA_PORT", port)                    # o "endereço padrão" deste teste
    monkeypatch.setattr(S, "wait_up", lambda *a, **k: False)
    monkeypatch.setattr(S, "start_local_ollama", fake_start)
    llm = LLM(_cfg(tmp_path, url))                                 # llm.retries=0
    try:
        assert llm.chat("generator", MSG, task="t")
        assert llm.chat("generator", MSG2, task="t")
        assert len(opened) == 1                                    # abriu uma vez só
    finally:
        for srv in opened:
            srv.shutdown()
            srv.server_close()


def test_autostart_can_be_turned_off(tmp_path, monkeypatch):
    from ragbench.core import server as S

    port = _free_port()
    opened = []
    monkeypatch.setattr(S, "OLLAMA_PORT", port)
    monkeypatch.setattr(S, "wait_up", lambda *a, **k: False)
    monkeypatch.setattr(S, "start_local_ollama", lambda *a, **k: opened.append(a) or "x")
    llm = LLM(_cfg(tmp_path, f"http://127.0.0.1:{port}", ["llm.autostart=false"]))
    with pytest.raises(LLMError) as e:
        llm.chat("generator", MSG, task="t")
    assert "não está respondendo" in str(e.value) and not opened


def test_ollama_that_cannot_be_opened_stops_with_the_message(tmp_path, monkeypatch):
    from ragbench.core import server as S

    port = _free_port()
    tried = []
    monkeypatch.setattr(S, "OLLAMA_PORT", port)
    monkeypatch.setattr(S, "wait_up", lambda *a, **k: False)
    monkeypatch.setattr(S, "start_local_ollama", lambda *a, **k: tried.append(a))      # devolve None: não abriu
    llm = LLM(_cfg(tmp_path, f"http://127.0.0.1:{port}"))
    for msg in (MSG, MSG2):
        with pytest.raises(LLMError) as e:
            llm.chat("generator", msg, task="t")
        assert "Abra o app do Ollama" in str(e.value)
    assert len(tried) == 1                                         # não fica tentando abrir a cada chamada


def test_server_that_was_just_starting_is_not_opened_again(tmp_path, monkeypatch):
    from ragbench.core import server as S

    port = _free_port()
    opened, live = [], []

    def came_up(base_url, seconds, step=1.0):                      # o servidor responde durante a espera curta
        live.append(fake_ollama.start(port)[0])
        return True

    monkeypatch.setattr(S, "OLLAMA_PORT", port)
    monkeypatch.setattr(S, "wait_up", came_up)
    monkeypatch.setattr(S, "start_local_ollama", lambda *a, **k: opened.append(a) or "x")
    try:
        assert LLM(_cfg(tmp_path, f"http://127.0.0.1:{port}")).chat("generator", MSG, task="t")
        assert not opened
    finally:
        for srv in live:
            srv.shutdown()
            srv.server_close()


def test_outage_after_the_first_answer_keeps_the_retries(tmp_path, monkeypatch):
    """Queda no meio da rodada (o servidor já tinha respondido): as tentativas com espera continuam valendo."""
    from ragbench.core import server as S

    srv, url = fake_ollama.start()
    naps, opened = [], []
    monkeypatch.setattr(S, "start_local_ollama", lambda *a, **k: opened.append(a) or "x")
    monkeypatch.setattr(S, "OLLAMA_PORT", int(url.rsplit(":", 1)[1]))
    llm = LLM(_cfg(tmp_path, url, ["llm.retries=2"]))
    assert llm.chat("generator", MSG, task="t")
    srv.shutdown()
    srv.server_close()
    monkeypatch.setattr("time.sleep", naps.append)
    with pytest.raises(LLMError) as e:
        llm.chat("generator", MSG2, task="t")
    assert "indisponível depois de 3 tentativas" in str(e.value)
    assert naps == [3, 10, 30] and not opened


def test_cached_round_needs_no_server(tmp_path, server, monkeypatch):
    """Com as respostas no cache, a rodada se repete sem servidor (é assim que se confere uma rodada antiga)."""
    from ragbench.core import server as S

    first = LLM(_cfg(tmp_path, server))
    text = first.chat("generator", MSG, task="t")
    first.close()
    monkeypatch.setattr(S, "is_up", lambda *a, **k: pytest.fail("não devia procurar o servidor"))
    again = LLM(_cfg(tmp_path, f"http://127.0.0.1:{_free_port()}"))
    assert again.chat("generator", MSG, task="t") == text


def test_server_helpers(server, monkeypatch):
    from ragbench.core import server as S

    assert S.is_up(server) and S.wait_up(server, 0)
    dead = f"http://127.0.0.1:{_free_port()}"
    assert not S.is_up(dead) and not S.wait_up(dead, 0)
    for url in ("http://localhost:11434", "http://127.0.0.1:11434/", "localhost:11434", "http://[::1]:11434"):
        assert S.is_local_ollama(url), url
    for url in ("http://localhost:8000",             # vLLM, llama.cpp...
                "http://192.168.0.10:11434",         # Ollama de outra máquina
                "https://api.openai.com/v1", "http://localhost:abc", ""):
        assert not S.is_local_ollama(url), url


def test_start_local_ollama(monkeypatch):
    from ragbench.core import server as S

    url = "http://localhost:11434"
    monkeypatch.setattr(S.shutil, "which", lambda name: None)
    assert S.start_local_ollama(url) is None                       # sem o comando ollama no PATH
    calls = []

    class FakePopen:
        def __init__(self, argv, **kw):
            calls.append((argv, kw))

    monkeypatch.setattr(S.shutil, "which", lambda name: "/opt/ollama")
    monkeypatch.setattr(S.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(S, "wait_up", lambda *a, **k: True)
    try:
        assert S.start_local_ollama(url) == "ollama serve"
        argv, kw = calls[0]
        assert argv == ["/opt/ollama", "serve"]
        assert kw["stdin"] == kw["stdout"] == kw["stderr"] == S.subprocess.DEVNULL
        monkeypatch.setattr(S, "wait_up", lambda *a, **k: False)
        assert S.start_local_ollama(url) is None                   # o processo abriu mas não respondeu

        def boom(*a, **k):
            raise OSError("não deu")

        monkeypatch.setattr(S.subprocess, "Popen", boom)
        assert S.start_local_ollama(url) is None
    finally:
        S._STARTED.clear()


# ----------------------------------------------------------------------------- indexação pela GPU do Ollama
def _docs(n=40):
    return [{"id": f"d{i}", "title": f"Title {i}", "text": f"paragraph about subject number {i} and its neighbours " * 4} for i in range(n)]


def test_index_embeddings_via_ollama(server):
    from ragbench.core.retrieval import DenseIndex

    docs = _docs()
    ref = DenseIndex("hash").build(docs, backend="st")
    assert ref.backend_used == "st-cpu"
    n_before = len(fake_ollama.LOG)
    idx = DenseIndex("hash").build(docs, backend="ollama", ollama_url=server)
    assert idx.backend_used == "ollama" and idx.E.shape == ref.E.shape
    assert float((idx.E * ref.E).sum(axis=1).min()) > 0.999          # os mesmos vetores que a CPU daria
    calls = [x for x in fake_ollama.LOG[n_before:] if x["path"] == "/api/embed"]
    assert calls and all(c["body"]["model"] == "hash" and isinstance(c["body"]["input"], list) for c in calls)
    # antes e depois de indexar, os LLMs são descarregados para liberar (e devolver) a memória da GPU
    unloads = [x for x in fake_ollama.LOG[n_before:] if x["path"] == "/api/generate" and x["body"].get("keep_alive") == 0]
    assert {u["body"]["model"] for u in unloads} >= {"qwen2.5:3b"}


def test_ollama_embeddings_are_checked_against_cpu(server):
    from ragbench.core.retrieval import DenseIndex

    texts = [f"{d['title']}\n{d['text']}" for d in _docs(12)]
    idx = DenseIndex("hash")
    assert idx._bulk_ollama(texts, server, "", check=True) is not None      # coincide com a CPU: aceito
    fake_ollama.EMBED_NOISE[0] = 0.8                                        # um modelo "parecido" que dá outros vetores
    try:
        assert DenseIndex("hash")._bulk_ollama(texts, server, "", check=True) is None      # recusado → o chamador usa a CPU
    finally:
        fake_ollama.EMBED_NOISE[0] = 0.0
    assert DenseIndex("hash")._bulk_ollama(texts, "http://127.0.0.1:9", "", check=True) is None   # Ollama fora do ar
    assert DenseIndex("intfloat/e5-base-v2")._bulk_ollama(texts, server, "") is None               # sem equivalente no Ollama
    # com o modelo de teste, "auto" nem tenta o Ollama (o smoke roda sem servidor nenhum)
    assert DenseIndex("hash").build(_docs(5), backend="auto", ollama_url="http://127.0.0.1:9").backend_used == "st-cpu"


def test_truncation_matches_sentence_transformers_cut():
    """O Ollama recebe o texto cortado no MESMO ponto em que o sentence-transformers cortaria (max_seq_length tokens)."""
    import re as _re

    from ragbench.core import retrieval as R

    class Tok:                      # tokenizer rápido de mentira: 1 palavra = 1 token, [CLS] e [SEP] contam no limite
        is_fast = True

        def __call__(self, texts, truncation=True, max_length=8, return_offsets_mapping=True):
            out = []
            for t in texts:
                offs = [(m.start(), m.end()) for m in _re.finditer(r"\S+", t)][: max_length - 2]
                out.append([(0, 0)] + offs + [(0, 0)])
            return {"offset_mapping": out}

    class Enc:
        tokenizer = Tok()

    idx = R.DenseIndex("hash", max_seq_length=8)
    idx._enc = lambda: Enc()
    long = "one two three four five six seven eight nine ten"
    assert idx._truncate_like_st([long, "short text", ""]) == ["one two three four five six", "short text", ""]
    idx._enc = lambda: object()     # sem tokenizer rápido: corte aproximado por palavras
    assert idx._truncate_like_st(["a " * 500])[0].count("a") == 200
