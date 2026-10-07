"""Servidor do painel (só biblioteca padrão) e controle da rodada de testes.

* Escuta apenas em 127.0.0.1. Requisições com outro ``Host`` são recusadas, e as que iniciam ou param uma
  rodada exigem o token que a própria página recebe — outra página aberta no navegador não consegue disparar
  comandos nesta máquina.
* Uma rodada é um subprocesso ``python -m ragbench <prepare|bench|report|doctor>`` com argumentos conferidos
  contra listas fechadas (perfis, datasets, artigos, tiers). Só uma rodada por vez: a GPU é uma só.
* Parar a rodada é seguro: tudo é retomável e as chamadas ao LLM ficam em cache.
"""
from __future__ import annotations

import codecs
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from ..config import PAPERS, PROFILES, load_config
from ..core.datasets import DATASET_NAMES
from . import data as D

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
KINDS = {"prepare": "preparar os dados", "bench": "rodar os testes", "report": "gerar os relatórios", "doctor": "verificar o ambiente"}
_NAME = re.compile(r"^[A-Za-z0-9_.\-/:@]+$")       # nomes de métodos
_DS = re.compile(r"^[A-Za-z0-9_\-]+$")              # nomes de datasets (viram nome de pasta: sem ponto nem barra)
_BAR = re.compile(r"^(?P<desc>.*?):\s*(?P<pct>\d+)%\|.*?\|\s*(?P<i>\d+)/(?P<n>\d+)\s*\[(?P<el>[^<\]]*)<(?P<eta>[^,\]]*),\s*(?P<rate>[^\]]*)\]")
_JOB = re.compile(r"▶ \[(\d+)/(\d+)\] tier (\d) · (\S+)")


# ============================================================================ a rodada (subprocesso)
def _kill_tree(proc: subprocess.Popen) -> None:
    """Encerra a rodada e tudo o que ela iniciou.

    No Windows, ``.venv\\Scripts\\python.exe`` é um lançador que inicia o Python de verdade como processo filho:
    ``taskkill /T`` encerra a árvore inteira, para não sobrar uma rodada órfã ocupando a GPU. O ``terminate()``
    fica como reserva (e é o caminho normal fora do Windows).
    """
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:  # noqa: BLE001 — sem taskkill, vale o terminate() abaixo
            pass
    try:
        if proc.poll() is None:
            proc.terminate()
    except Exception:  # noqa: BLE001 — o processo terminou entre o teste e a chamada
        pass


class Runner:
    def __init__(self, root: str, overrides: list[str]):
        self.root, self.overrides = root, list(overrides or [])
        self.lock = threading.Lock()
        self._log_lock = threading.Lock()         # o log é escrito pela thread do subprocesso e lido pelas requisições
        self.proc: subprocess.Popen | None = None
        self.lines: deque = deque(maxlen=800)
        self.info: dict = {"state": "idle"}
        self.bar = ""
        self.job: dict | None = None
        self.seq = 0

    # ------------------------------------------------------------------ comando
    def build_argv(self, kind: str, profile: str, datasets=None, papers=None, tiers=None, methods=None) -> list[str]:
        if kind not in KINDS:
            raise ValueError(f"ação desconhecida: {kind}")
        if profile not in PROFILES:
            raise ValueError(f"perfil desconhecido: {profile}")
        datasets = [str(d) for d in (datasets or [])]
        papers = [str(p) for p in (papers or [])]
        tiers = [int(t) for t in (tiers or [])]
        methods = [str(m) for m in (methods or [])]
        if any(d not in DATASET_NAMES for d in datasets):
            raise ValueError("dataset desconhecido")
        if any(p not in PAPERS for p in papers):
            raise ValueError("artigo desconhecido")
        if any(t not in (1, 2, 3) for t in tiers):
            raise ValueError("tier desconhecido")
        if any(not _NAME.match(m) for m in methods):
            raise ValueError("nome de método inválido")
        argv = [sys.executable, "-m", "ragbench", kind, "--profile", profile, "--root", self.root]
        for ov in self.overrides:
            argv += ["--set", ov]
        if kind in ("prepare", "bench") and datasets:
            argv += ["--datasets", *datasets]
        if kind == "bench":
            if papers and set(papers) != set(PAPERS):
                argv += ["--paper", *papers]
            if tiers:
                argv += ["--tier", *map(str, sorted(set(tiers)))]
            if methods:
                argv += ["--methods", *methods]
        return argv

    def start(self, kind: str, profile: str, **kw) -> dict:
        argv = self.build_argv(kind, profile, **kw)
        with self.lock:
            if self.proc is not None and self.proc.poll() is None:
                raise RuntimeError("já existe uma rodada em andamento — espere terminar ou clique em Parar")
            env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
            pkg_parent = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env["PYTHONPATH"] = pkg_parent + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            with self._log_lock:
                self.lines.clear()
            self.bar, self.job = "", None
            self.proc = subprocess.Popen(argv, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL)
            shown = " ".join(["python", *argv[1:]])
            self.info = {"state": "running", "kind": kind, "label": KINDS[kind], "profile": profile, "cmd": shown,
                         "started": time.time(), "ended": None, "rc": None}
            self._push(f"$ {shown}")
            threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        return self.status()

    def _push(self, line: str) -> None:
        with self._log_lock:
            self.lines.append(line)
            self.seq += 1
        m = _JOB.search(line)
        if m:
            self.job = {"i": int(m.group(1)), "n": int(m.group(2)), "tier": int(m.group(3)), "key": m.group(4)}

    def _pump(self, proc: subprocess.Popen) -> None:
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        cur = ""
        while True:
            chunk = proc.stdout.read1(4096) if hasattr(proc.stdout, "read1") else proc.stdout.read(4096)
            if not chunk:
                break
            for part in re.split(r"(\r\n|\n|\r)", dec.decode(chunk)):
                if part in ("\n", "\r\n"):
                    if cur.strip():
                        self._push(cur.rstrip())
                    cur = ""
                elif part == "\r":                  # barra de progresso do tqdm: a linha é reescrita
                    if cur.strip():
                        self.bar = cur.strip()
                        self.seq += 1
                    cur = ""
                else:
                    cur += part
        if cur.strip():
            self._push(cur.rstrip())
        rc = proc.wait()
        with self.lock:
            if self.proc is proc:
                stopped = self.info.get("state") == "stopping"
                self.info.update(state="stopped" if stopped else ("done" if rc == 0 else "failed"), rc=rc, ended=time.time())
                self.bar = ""
                self._push("— rodada interrompida; rode de novo para continuar de onde parou —" if stopped
                           else f"— terminou (código {rc}) —")

    def stop(self) -> dict:
        with self.lock:
            proc = self.proc if (self.proc is not None and self.proc.poll() is None) else None
            if proc is not None:
                self.info["state"] = "stopping"
        if proc is not None:
            _kill_tree(proc)
        return self.status()

    def status(self) -> dict:
        bar = None
        m = _BAR.match(self.bar or "")
        if m:
            bar = {**m.groupdict(), "pct": int(m.group("pct")), "i": int(m.group("i")), "n": int(m.group("n"))}
        with self._log_lock:
            lines = list(self.lines)[-250:]
        return {**self.info, "seq": self.seq, "job": self.job, "bar": bar, "bar_raw": self.bar if not bar else "",
                "log": lines, "kinds": KINDS}

    def shutdown(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            _kill_tree(self.proc)


# ============================================================================ aplicação
class App:
    def __init__(self, root: str, profile: str, overrides: list[str] | None = None, readonly: bool = False):
        self.root, self.profile, self.overrides = os.path.abspath(root), profile, list(overrides or [])
        self.readonly = readonly
        self.token = secrets.token_urlsafe(24)
        self.runner = Runner(self.root, self.overrides)
        self._cfgs: dict = {}

    def cfg(self, profile: str | None) -> dict:
        profile = profile or self.profile
        if profile not in PROFILES:
            raise ValueError(f"perfil desconhecido: {profile}")
        if profile not in self._cfgs:
            self._cfgs[profile] = load_config(profile, self.overrides, self.root)
        return self._cfgs[profile]

    def boot(self) -> dict:
        return {"token": self.token, "profile": self.profile, "readonly": self.readonly, "export": False,
                "root": self.root, "profiles": D.profiles(self.root, self.overrides)}

    # ------------------------------------------------------------------ GET
    def get(self, path: str, q: dict):
        def one(k, d=""):
            return (q.get(k) or [d])[0]

        def many(k):
            return [x for v in q.get(k, []) for x in v.split(",") if x]

        def num(k, d=0):
            try:
                return max(0, int(one(k, str(d))))
            except ValueError:
                return d

        cfg = self.cfg(one("profile") or None)
        ds = one("dataset")
        if ds and not _DS.match(ds):
            raise ValueError("dataset inválido")
        if path == "/api/profiles":
            return D.profiles(self.root, self.overrides)
        if path == "/api/state":
            return D.state(cfg)
        if path == "/api/matrix":
            return D.matrix(cfg, ds)
        if path == "/api/test":
            out = D.test_detail(cfg, ds, one("id"), many("methods") or None)
            if out is None:
                raise KeyError("teste não encontrado")
            return out
        if path == "/api/questions":
            return D.questions(cfg, ds, one("split", "eval"), one("q"), num("offset"), min(5000, num("limit", 5000) or 5000))
        if path == "/api/corpus":
            return D.corpus_search(cfg, ds, one("q"), num("offset"), min(100, num("limit", 30) or 30))
        if path == "/api/errorbench":
            return D.errorbench(cfg, one("tag"), num("offset"), min(50, num("limit", 20) or 20), one("label"), one("show"))
        if path == "/api/reports":
            return D.reports(cfg)
        if path == "/api/report":
            text = D.report_text(cfg, one("path"))
            if text is None:
                raise KeyError("relatório não encontrado")
            return {"path": one("path"), "text": text}
        if path == "/api/plan":
            return D.plan(cfg, many("datasets"), many("papers"), [int(t) for t in many("tiers") if t.isdigit()])
        if path == "/api/run":
            return self.runner.status()
        raise KeyError(path)

    # ------------------------------------------------------------------ POST
    def post(self, path: str, body: dict):
        if self.readonly:
            raise PermissionError("painel aberto em modo somente leitura")
        if path == "/api/run":
            return self.runner.start(str(body.get("kind")), str(body.get("profile") or self.profile),
                                     datasets=body.get("datasets"), papers=body.get("papers"), tiers=body.get("tiers"),
                                     methods=body.get("methods"))
        if path == "/api/stop":
            return self.runner.stop()
        raise KeyError(path)


def _read_static(name: str) -> str:
    with open(os.path.join(STATIC, name), encoding="utf-8") as f:
        return f.read()


def render_page(boot: dict, embedded: dict | None = None) -> str:
    """A página inteira num HTML só (CSS e JS embutidos; nada vem da internet)."""
    html = _read_static("index.html")
    payload = json.dumps(boot, ensure_ascii=False).replace("</", "<\\/")
    data = json.dumps(embedded, ensure_ascii=False).replace("</", "<\\/") if embedded is not None else "null"
    return (html.replace("/*__CSS__*/", _read_static("app.css"))
            .replace("/*__BOOT__*/", f"window.RB_BOOT = {payload};\nwindow.RB_DATA = {data};")
            .replace("/*__JS__*/", _read_static("app.js")))


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        server_version = "ragbench-ui"

        def log_message(self, *a):  # silêncio: o console fica para os avisos importantes
            pass

        def _host_ok(self) -> bool:
            host = (self.headers.get("Host") or "").split(":")[0].lower()
            return host in ("127.0.0.1", "localhost", "[::1]")

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, ensure_ascii=False, allow_nan=False, default=_default).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _fail(self, e: Exception) -> None:
            code = 404 if isinstance(e, (KeyError, FileNotFoundError)) else 403 if isinstance(e, PermissionError) else \
                409 if isinstance(e, RuntimeError) else 400 if isinstance(e, ValueError) else 500
            self._json(code, {"error": str(e).strip("'\"") or type(e).__name__})

        def do_GET(self):  # noqa: N802
            if not self._host_ok():
                return self._send(403, b"host nao permitido", "text/plain")
            u = urlparse(self.path)
            try:
                if u.path in ("/", "/index.html"):
                    return self._send(200, render_page(app.boot()).encode("utf-8"), "text/html; charset=utf-8")
                if u.path == "/api/file":
                    q = parse_qs(u.query)
                    full = D.safe_result_file(app.cfg((q.get("profile") or [None])[0]), (q.get("path") or [""])[0])
                    if not full:
                        raise KeyError("arquivo não encontrado")
                    with open(full, "rb") as f:
                        return self._send(200, f.read(), "image/png")
                if u.path.startswith("/api/"):
                    return self._json(200, _clean(app.get(u.path, parse_qs(u.query))))
                self._send(404, b"nao encontrado", "text/plain")
            except Exception as e:  # noqa: BLE001
                self._fail(e)

        def do_POST(self):  # noqa: N802
            if not self._host_ok():
                return self._json(403, {"error": "host não permitido"})
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).hostname not in ("127.0.0.1", "localhost", "::1"):
                return self._json(403, {"error": "origem não permitida"})
            if not secrets.compare_digest(self.headers.get("X-RagBench-Token") or "", app.token):
                return self._json(403, {"error": "token inválido — recarregue a página"})
            try:
                n = min(int(self.headers.get("Content-Length") or 0), 1 << 20)
                body = json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}
                if not isinstance(body, dict):
                    raise ValueError("corpo inválido")
                self._json(200, _clean(app.post(urlparse(self.path).path, body)))
            except Exception as e:  # noqa: BLE001
                self._fail(e)

    return Handler


def _default(o):
    return str(o)


def _clean(o):
    """NaN e infinito não existem em JSON: viram ``null``."""
    if isinstance(o, float):
        return o if o == o and o not in (float("inf"), float("-inf")) else None
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # Porta ocupada tem de dar erro, para o make_server passar para a seguinte. No Windows, SO_REUSEADDR deixaria um
    # segundo painel ouvir na MESMA porta do primeiro (e os dois se atrapalhariam); SO_REUSEPORT faz o mesmo no
    # Linux/macOS. Fora do Windows, SO_REUSEADDR só permite reabrir a porta logo depois de fechar.
    allow_reuse_address = os.name != "nt"
    allow_reuse_port = False


def make_server(app: App, port: int = 8765, tries: int = 20) -> ThreadingHTTPServer:
    last = None
    for p in range(port, port + tries):
        try:
            return _Server(("127.0.0.1", p), make_handler(app))
        except OSError as e:           # porta ocupada: tenta a próxima
            last = e
    raise RuntimeError(f"nenhuma porta livre entre {port} e {port + tries - 1}: {last}")


def serve(root: str, profile: str, overrides: list[str] | None = None, port: int = 8765, open_browser: bool = True) -> None:
    app = App(root, profile, overrides)
    srv = make_server(app, port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"\n  ragbench · painel aberto em  {url}\n  perfil: {profile} · pasta: {app.root}\n"
          f"  Deixe esta janela aberta enquanto usa o painel. Para fechar: Ctrl+C.\n", flush=True)
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\npainel fechado.")
    finally:
        app.runner.shutdown()
        srv.server_close()


# ============================================================================ exportação (um HTML, sem servidor)
def data_key(route: str, **params) -> str:
    """Chave de um dado embutido — a mesma que ``keyOf`` monta em ``app.js`` (parâmetros em ordem alfabética)."""
    ks = sorted(k for k, v in params.items() if v not in (None, ""))
    return route + "?" + "&".join(f"{k}={params[k]}" for k in ks)


def export_html(root: str, profile: str, out_path: str, overrides: list[str] | None = None,
                methods: list[str] | None = None, max_mb: float = 40.0) -> str:
    """Grava um HTML único com os resultados do perfil. Abre com dois cliques, em qualquer máquina, sem servidor.

    Vão: o estado, a matriz de todos os datasets (todos os métodos), as perguntas de avaliação, os relatórios e
    o detalhe de cada teste para os métodos escolhidos (padrão: os de ``compare.default_methods`` e as bases
    deles). O corpus e o RAG-Error Bench ficam de fora.
    """
    app = App(root, profile, overrides, readonly=True)
    cfg = app.cfg(profile)
    st = D.state(cfg)
    # o arquivo pode ser mandado para outra pessoa: os caminhos completos desta máquina ficam de fora
    st = {**st, "root": os.path.basename(app.root), "results": os.path.relpath(st["results"], app.root).replace(os.sep, "/")}
    emb: dict = {data_key("/api/state", profile=profile): st, data_key("/api/reports", profile=profile): D.reports(cfg)}
    for r in D.reports(cfg):
        emb[data_key("/api/report", profile=profile, path=r["path"])] = {"path": r["path"], "text": D.report_text(cfg, r["path"])}
    n_tests = 0
    for d in st["datasets"]:
        ds = d["key"]
        if d["prepared"]:
            emb[data_key("/api/questions", profile=profile, dataset=ds, split="eval")] = D.questions(cfg, ds, "eval", "", 0, 5000)
        if not d["n_methods"]:
            continue
        m = D.matrix(cfg, ds)
        emb[data_key("/api/matrix", profile=profile, dataset=ds)] = m
        keep = set(methods or [])
        if not keep:
            keep = set(m["defaults"])
            keep |= {x["base"] for x in m["methods"] if x["key"] in keep and x["base"]}
        keep &= set(m["cells"])
        if not keep:                        # nenhum dos métodos pedidos rodou neste dataset: só a matriz
            continue
        for q in m["questions"]:
            det = D.test_detail(cfg, ds, q["id"], sorted(keep))
            if det:
                emb[data_key("/api/test", profile=profile, dataset=ds, id=q["id"])] = det
                n_tests += 1
    boot = {**app.boot(), "export": True, "readonly": True, "token": "", "root": os.path.basename(app.root),
            "exported": time.strftime("%Y-%m-%d %H:%M"),
            "profiles": [p for p in D.profiles(app.root, overrides) if p["name"] == profile]}
    html = render_page(boot, _clean(emb))
    size = len(html.encode("utf-8")) / 2 ** 20
    if size > max_mb:
        raise RuntimeError(f"o arquivo ficaria com {size:.0f} MB; escolha menos métodos com --methods")
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return f"{out_path} ({size:.1f} MB, {n_tests} testes com detalhe)"
