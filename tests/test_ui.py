"""Painel: dados que a página recebe, servidor (rotas, proteção das ações), controle da rodada e exportação."""
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ragbench.ui import data as D  # noqa: E402
from ragbench.ui.server import App, Runner, data_key, export_html, make_server, render_page  # noqa: E402


# ============================================================================ dados
def test_state_and_profiles(smoke_cfg, smoke_root):
    ps = {p["name"]: p for p in D.profiles(smoke_root)}
    assert ps["smoke"]["n_files"] > 0 and ps["quick"]["n_files"] == 0 and ps["smoke"]["datasets"] == ["toy_multihop", "toy_single"]
    assert D.latest_profile(smoke_root) == "smoke"                     # o único com resultado
    assert D.latest_profile(os.path.join(smoke_root, "vazia"), default="quick") == "quick"
    st = D.state(smoke_cfg)
    ds = {d["key"]: d for d in st["datasets"]}
    assert ds["toy_multihop"]["prepared"] and ds["toy_multihop"]["n_methods"] >= 6 and ds["toy_multihop"]["compare"]
    assert set(ds["toy_multihop"]["papers"]) == {"ragcritic", "dynakrag"} and ds["toy_single"]["papers"] == ["ragcritic"]
    assert set(st["papers"]) == set(st["n_eval"]) == set(st["retrieval"]) == {"ragcritic", "dynakrag"}
    assert set(st["models"]) == {"generator", "critic", "planner", "judge", "summarizer"}
    assert st["compare"]["metric"] == "em" and st["compare"]["thr"] == 0.5 and "rag-critic-3b" in st["errorbench"]
    assert all(v["state"] == "done" for v in st["jobs"].values())
    json.dumps(st)                                                     # tudo o que vai para a página é serializável


def test_jobs_of_a_removed_paper_are_hidden(tmp_path):
    """Um ``status.json`` antigo, com jobs de um artigo que saiu do projeto, não aparece no painel nem no índice."""
    from ragbench.config import load_config
    from ragbench.report import build_all

    cfg = load_config("smoke", [], root=str(tmp_path))
    res = cfg["paths"]["results"]
    os.makedirs(res)
    with open(os.path.join(res, "status.json"), "w", encoding="utf-8") as f:
        json.dump({"ragcritic/toy_multihop/ragcritic": {"state": "done", "tier": 1, "secs": 60.0},
                   "doctorrag/toy_multihop/react": {"state": "done", "tier": 1, "secs": 3600.0},
                   "doctorrag/toy_multihop/react-drrag": {"state": "failed", "tier": 1, "error": "x"}}, f)
    assert list(D.state(cfg)["jobs"]) == ["ragcritic/toy_multihop/ragcritic"]
    index = open(build_all(cfg)["index"], encoding="utf-8").read()
    assert "`ragcritic/toy_multihop/ragcritic`" in index and "doctorrag" not in index
    assert "registrado: 0.02 h" in index                               # 60 s: o tempo do artigo que saiu não entra


def test_matrix_is_cached_and_json_safe(smoke_cfg):
    m1 = D.matrix(smoke_cfg, "toy_multihop")
    assert D.matrix(smoke_cfg, "toy_multihop") is m1                   # sem mudança no disco: o mesmo objeto
    assert m1["defaults"] == ["ragcritic/standard_rag", "ragcritic/ragcritic", "dynakrag/dynakrag"]
    json.dumps(m1, allow_nan=False)
    assert D.matrix(smoke_cfg, "nao_existe")["methods"] == []


def test_test_detail_has_steps_for_every_method(smoke_cfg):
    m = D.matrix(smoke_cfg, "toy_multihop")
    qid = m["questions"][0]["id"]
    d = D.test_detail(smoke_cfg, "toy_multihop", qid)
    assert d["question"] and d["answers"] and d["support"] and d["support_docs"] and d["has_corpus"]
    assert set(d["methods"]) == set(m["cells"])
    for key, mm in d["methods"].items():
        assert mm["steps"], key
        assert all(s["k"] and s["t"] for s in mm["steps"]), key
        assert not any(s["t"].startswith("Não foi possível") for s in mm["steps"]), key     # nenhum formato quebrou a montagem
        assert all(k in d["docs"] for k in mm["docs"]), key
        assert mm["p"] == m["cells"][key][0]["p"] and mm["f"] == m["cells"][key][0]["f"]      # mesma célula da matriz
    kinds = {k: [s["k"] for s in v["steps"]] for k, v in d["methods"].items()}
    assert kinds["ragcritic/standard_rag"] == ["busca", "resposta"]
    assert "critico" in kinds["ragcritic/ragcritic"] and "acao" in kinds["dynakrag/dynakrag"]
    assert all(k[-1] == "resposta" for k in kinds.values())                                   # todo passo a passo termina na resposta
    assert all(key.split("/")[0] in ("ragcritic", "dynakrag") for key in kinds)
    sup = [x for x in d["docs"].values() if x["sup"]]
    assert sup and all(x["title"] in d["support"] for x in sup)
    only = D.test_detail(smoke_cfg, "toy_multihop", qid, ["dynakrag/dynakrag"])
    assert list(only["methods"]) == ["dynakrag/dynakrag"]
    assert D.test_detail(smoke_cfg, "toy_multihop", "nao-existe") is None
    json.dumps(d, allow_nan=False)


def test_data_browsers(smoke_cfg):
    q = D.questions(smoke_cfg, "toy_multihop", "eval", "", 0, 5000)
    assert q["total"] == 12 and len(q["items"]) == 12 and all(x["used"] for x in q["items"]) and q["items"][0]["sup"]
    assert D.questions(smoke_cfg, "toy_multihop", "train")["split"] == "train"
    assert D.questions(smoke_cfg, "toy_multihop", "../../x")["split"] == "eval"              # split fora da lista → avaliação
    word = q["items"][0]["q"].split()[-1].strip("?")
    assert 0 < D.questions(smoke_cfg, "toy_multihop", "eval", word)["matched"] <= 12
    c = D.corpus_search(smoke_cfg, "toy_multihop", "", 0, 10)
    assert c["total"] > 0 and len(c["items"]) == 10 and any(x["sup"] for x in D.corpus_search(smoke_cfg, "toy_multihop", "", 0, 100)["items"])
    assert D.corpus_search(smoke_cfg, "toy_multihop", "zzzz-nao-existe")["matched"] == 0
    assert D.corpus_search(smoke_cfg, "sem_dados")["missing"] is True
    eb = D.errorbench(smoke_cfg)
    assert eb["tag"] == "rag-critic-3b" and eb["summary"]["n"] == 12 and len(eb["items"]) == 12 and eb["has_items"]
    it = eb["items"][0]
    assert it["question"] and len(it["passages"]) == 5 and it["original"] and it["swapped"] and it["original"]["indent"] != it["swapped"]["indent"]
    assert D.errorbench(smoke_cfg, label="Error")["matched"] == 6
    assert all(x["original"]["judgement"] != x["label"] for x in D.errorbench(smoke_cfg, show="errou")["items"])


def test_reports_listing_is_a_whitelist(smoke_cfg):
    from ragbench.report import build_all

    build_all(smoke_cfg)
    res = smoke_cfg["paths"]["results"]
    old = os.path.join(res, "doctorrag", "REPORT.md")                 # relatório de um artigo que saiu do projeto
    os.makedirs(os.path.dirname(old), exist_ok=True)
    with open(old, "w", encoding="utf-8") as f:
        f.write("# Doctor-RAG isolado\n")
    try:
        paths = [r["path"] for r in D.reports(smoke_cfg)]
        assert paths == ["REPORT.md", "TABELAS.md", "FALHAS.md", "COMPARACAO.md", "ragcritic/REPORT.md", "dynakrag/REPORT.md"]
        assert D.report_text(smoke_cfg, "doctorrag/REPORT.md") is None               # existe no disco, mas não está na lista
    finally:
        os.remove(old)
        os.rmdir(os.path.dirname(old))
    assert D.report_text(smoke_cfg, "FALHAS.md").startswith("# Análise de falhas")
    assert D.report_text(smoke_cfg, "../../etc/passwd") is None and D.report_text(smoke_cfg, "status.json") is None
    assert D.safe_result_file(smoke_cfg, "../../../etc/passwd") is None
    assert D.safe_result_file(smoke_cfg, "status.json") is None                      # só figuras .png
    assert "Falha ao gerar" not in open(os.path.join(smoke_cfg["paths"]["results"], "REPORT.md"), encoding="utf-8").read()


def test_plan_estimates(smoke_cfg):
    p = D.plan(smoke_cfg, ["toy_multihop"], None, [1])
    keys = [j["key"] for j in p["jobs"]]
    assert "ragcritic/toy_multihop/ragcritic" in keys and "dynakrag/toy_multihop/train_controller" in keys
    assert {j["paper"] for j in p["jobs"]} == {"ragcritic", "dynakrag"}
    assert p["n_todo"] == 0 and p["calls"] == 0 and p["missing_data"] == []           # o tier 1 já rodou na fixture
    p3 = D.plan(smoke_cfg, ["toy_multihop"], ["dynakrag"], [3])
    assert p3["n_todo"] == p3["n"] > 0 and p3["calls"] > 0 and all(j["paper"] == "dynakrag" for j in p3["jobs"])
    assert D.plan(smoke_cfg, ["toy_multihop"], ["doctorrag"], [1])["n"] == 0          # artigo que não existe: nada a rodar
    assert D.plan(smoke_cfg, ["hotpotqa"], ["dynakrag"], [1])["missing_data"] == []   # fora dos datasets do perfil: nada a rodar


# ============================================================================ servidor
@pytest.fixture(scope="module")
def server(smoke_root):
    app = App(smoke_root, "smoke")
    srv = make_server(app, port=8791)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield app, f"http://127.0.0.1:{srv.server_address[1]}"
    app.runner.shutdown()
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8"), dict(e.headers)


def _post(url, body, headers=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8") or "{}")


def test_page_is_self_contained(server):
    app, url = server
    code, html, headers = _get(url + "/")
    assert code == 200 and app.token in html and "window.RB_BOOT" in html
    assert "/*__CSS__*/" not in html and "/*__JS__*/" not in html and "/*__BOOT__*/" not in html
    # nada vem da internet: nenhum script, folha de estilo, fonte ou imagem externos
    assert not re.search(r"<(script|link|img|iframe)[^>]+(src|href)\s*=\s*[\"']https?:", html, re.I)
    assert "@import" not in html and "url(http" not in html
    assert headers.get("Cache-Control") == "no-store" and headers.get("X-Content-Type-Options") == "nosniff"


def test_api_routes(server):
    _, url = server
    code, body, _ = _get(url + "/api/state?profile=smoke")
    assert code == 200 and json.loads(body)["profile"] == "smoke"
    m = json.loads(_get(url + "/api/matrix?profile=smoke&dataset=toy_multihop")[1])
    qid = m["questions"][0]["id"]
    d = json.loads(_get(url + f"/api/test?profile=smoke&dataset=toy_multihop&id={qid}")[1])
    assert d["id"] == qid and d["methods"]
    assert json.loads(_get(url + "/api/questions?profile=smoke&dataset=toy_multihop")[1])["total"] == 12
    assert json.loads(_get(url + "/api/corpus?profile=smoke&dataset=toy_multihop&q=&limit=3")[1])["items"]
    assert json.loads(_get(url + "/api/errorbench?profile=smoke")[1])["tags"]
    assert json.loads(_get(url + "/api/plan?profile=smoke&datasets=toy_multihop&tiers=1")[1])["n"] > 0
    assert {p["name"] for p in json.loads(_get(url + "/api/profiles")[1])} >= {"smoke", "mini", "quick", "standard"}
    assert json.loads(_get(url + "/api/run")[1])["state"] in ("idle", "done", "failed", "stopped", "running")


def test_bad_requests_are_refused(server):
    _, url = server
    assert _get(url + "/api/matrix?profile=smoke&dataset=../../etc")[0] == 400        # nome de dataset vira pasta: sem ponto nem barra
    assert _get(url + "/api/state?profile=nao_existe")[0] == 400
    assert _get(url + "/api/test?profile=smoke&dataset=toy_multihop&id=nao-existe")[0] == 404
    assert _get(url + "/api/report?profile=smoke&path=../status.json")[0] == 404
    assert _get(url + "/api/file?profile=smoke&path=../../../etc/passwd")[0] == 404
    assert _get(url + "/api/nao-existe")[0] == 404 and _get(url + "/qualquer")[0] == 404
    assert _get(url + "/api/state", {"Host": "evil.example"})[0] == 403                # proteção contra DNS rebinding
    assert _get(url + "/", {"Host": "evil.example:8791"})[0] == 403


def test_run_needs_token_and_valid_arguments(server):
    app, url = server
    assert _post(url + "/api/run", {"kind": "doctor", "profile": "smoke"})[0] == 403                      # sem token
    assert _post(url + "/api/run", {"kind": "doctor"}, {"X-RagBench-Token": "errado"})[0] == 403
    tok = {"X-RagBench-Token": app.token}
    assert _post(url + "/api/run", {"kind": "doctor"}, {**tok, "Origin": "http://evil.example"})[0] == 403   # outra página chamando
    assert _post(url + "/api/run", {"kind": "rm -rf", "profile": "smoke"}, tok)[0] == 400
    assert _post(url + "/api/run", {"kind": "bench", "profile": "x; rm -rf /"}, tok)[0] == 400
    assert _post(url + "/api/run", {"kind": "bench", "profile": "smoke", "datasets": ["--root", "/tmp"]}, tok)[0] == 400
    assert _post(url + "/api/run", {"kind": "bench", "profile": "smoke", "methods": ["a b; c"]}, tok)[0] == 400
    assert _post(url + "/api/run", {"kind": "bench", "profile": "smoke", "tiers": [9]}, tok)[0] == 400
    assert _post(url + "/api/outra", {}, tok)[0] == 404
    # uma ação de verdade: o "doctor" com o LLM falso termina sozinho e o log chega pela API
    code, st = _post(url + "/api/run", {"kind": "doctor", "profile": "smoke"}, tok)
    assert code == 200 and st["state"] == "running" and st["cmd"].startswith("python -m ragbench doctor --profile smoke")
    assert _post(url + "/api/run", {"kind": "doctor", "profile": "smoke"}, tok)[0] in (409, 200)           # uma rodada por vez
    t0 = time.time()
    while time.time() - t0 < 90:
        st = json.loads(_get(url + "/api/run")[1])
        if st["state"] != "running":
            break
        time.sleep(0.3)
    assert st["state"] == "done" and st["rc"] == 0 and any("backend mock" in ln for ln in st["log"])


def test_runner_argv_and_stop(smoke_root):
    r = Runner(smoke_root, ["llm.backend=mock"])
    argv = r.build_argv("bench", "mini", datasets=["hotpotqa", "2wiki"], papers=["ragcritic"], tiers=[2, 1])
    assert argv[1:6] == ["-m", "ragbench", "bench", "--profile", "mini"] and argv[6:8] == ["--root", smoke_root]
    assert argv[8:10] == ["--set", "llm.backend=mock"]
    tail = argv[10:]
    assert tail == ["--datasets", "hotpotqa", "2wiki", "--paper", "ragcritic", "--tier", "1", "2"]
    assert "--paper" not in r.build_argv("bench", "mini", papers=["ragcritic", "dynakrag"])                 # os dois = sem filtro
    assert "--tier" not in r.build_argv("prepare", "mini", datasets=["nq"], tiers=[1])                       # só o bench tem tier
    assert r.build_argv("report", "quick")[3:6] == ["report", "--profile", "quick"]
    for bad in ({"datasets": ["../x"]}, {"papers": ["outro"]}, {"papers": ["doctorrag"]}, {"tiers": [0]}, {"methods": ["$(x)"]}):
        with pytest.raises(ValueError):
            r.build_argv("bench", "mini", **bad)
    with pytest.raises(ValueError):
        r.build_argv("bench", "nao_existe")
    assert r.status()["state"] == "idle" and r.stop()["state"] == "idle"                                    # parar sem rodada: nada acontece


def test_readonly_app_refuses_actions(smoke_root):
    app = App(smoke_root, "smoke", readonly=True)
    with pytest.raises(PermissionError):
        app.post("/api/run", {"kind": "doctor"})


# ============================================================================ exportação
def test_export_single_file(smoke_root, smoke_cfg, tmp_path):
    out = str(tmp_path / "painel.html")
    msg = export_html(smoke_root, "smoke", out)
    assert out in msg and os.path.exists(out)
    html = open(out, encoding="utf-8").read()
    assert not re.search(r"<(script|link|img|iframe)[^>]+(src|href)\s*=\s*[\"']https?:", html, re.I)
    m = re.search(r"window\.RB_BOOT = (.*);\nwindow\.RB_DATA = (.*);</script>", html)      # cada JSON ocupa uma linha só
    boot, data = json.loads(m.group(1).replace("<\\/", "</")), json.loads(m.group(2).replace("<\\/", "</"))
    assert boot["export"] and boot["readonly"] and boot["token"] == "" and [p["name"] for p in boot["profiles"]] == ["smoke"]
    # o arquivo pode ser mandado para outra pessoa: o caminho completo desta máquina não vai junto
    assert boot["root"] == os.path.basename(smoke_root) and os.path.dirname(smoke_root) not in html
    mk = data_key("/api/matrix", profile="smoke", dataset="toy_multihop")
    assert mk == "/api/matrix?dataset=toy_multihop&profile=smoke" and mk in data
    assert data_key("/api/state", profile="smoke") in data and data_key("/api/questions", profile="smoke", dataset="toy_multihop", split="eval") in data
    qid = data[mk]["questions"][0]["id"]
    det = data[data_key("/api/test", profile="smoke", dataset="toy_multihop", id=qid)]
    best_k = next(x["base"] for x in data[mk]["methods"] if x["key"] == "dynakrag/dynakrag")
    assert set(det["methods"]) == {"ragcritic/standard_rag", "ragcritic/ragcritic", "dynakrag/dynakrag", best_k}
    assert "ragcritic/self_refine" not in det["methods"]               # o detalhe só vai para os métodos escolhidos
    only = str(tmp_path / "so_um.html")
    export_html(smoke_root, "smoke", only, methods=["dynakrag/dynakrag"])
    assert os.path.getsize(only) < os.path.getsize(out)
    # um método que não existe (nome antigo, erro de digitação) não vira "todos os métodos": vai só a matriz
    none = str(tmp_path / "nenhum.html")
    assert "0 testes com detalhe" in export_html(smoke_root, "smoke", none, methods=["doctorrag/react-drrag_blind"])
    assert os.path.getsize(none) < os.path.getsize(only)
    with pytest.raises(RuntimeError):
        export_html(smoke_root, "smoke", str(tmp_path / "x.html"), max_mb=0.01)


def test_render_page_escapes_script_end():
    html = render_page({"token": "t", "x": "</script><script>alert(1)</script>"}, {"k": "</script>"})
    assert "</script><script>alert(1)" not in html and html.count("</script>") == 2      # só os dois fechamentos do modelo
