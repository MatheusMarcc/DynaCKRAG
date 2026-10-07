"""Cliente de LLM: Ollama (padrão), API compatível com OpenAI (vLLM, llama.cpp, APIs) ou mock.

O que ele resolve:
  * **cache em disco (SQLite)**: com temperatura 0, repetir um experimento não gasta nada; só as
    chamadas novas vão ao modelo. Também é o que torna a rodada *retomável* depois de uma interrupção.
  * **contabilidade por pergunta**: ``with llm.track() as u:`` soma chamadas e tokens (inclusive
    das chamadas vindas do cache, com a latência original), para as colunas Tok./LLM calls.
  * **template ChatML cru** (``template="chatml"``): monta o prompt à mão e chama ``/api/generate``
    com ``raw=true``. Usado no crítico RAG-Critic-3B para garantir o formato do demo oficial
    (system padrão do Qwen + prompt como mensagem de usuário), sem depender do template que o
    Ollama infere do GGUF.
  * **logprobs** (FLARE): pedidos ao Ollama com ``logprobs=true`` (Ollama recente) ou à API OpenAI.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
import time
import weakref
from collections import defaultdict

from .io import ensure_dir, get_logger
from .tokens import count_messages, count_tokens

QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


class LLMError(RuntimeError):
    pass


def render_chatml(messages: list[dict], default_system: str | None = QWEN_DEFAULT_SYSTEM) -> str:
    msgs = list(messages)
    if default_system is not None and (not msgs or msgs[0].get("role") != "system"):
        msgs = [{"role": "system", "content": default_system}] + msgs
    out = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
    return out + "<|im_start|>assistant\n"


class LLM:
    def __init__(self, cfg: dict):
        lc = cfg["llm"]
        self.cfg = cfg
        self.backend = lc.get("backend", "ollama")
        self.base_url = str(lc.get("base_url", "http://localhost:11434")).rstrip("/")
        self.api_key = lc.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self.num_ctx = int(lc.get("num_ctx", 8192))
        self.seed = int(cfg.get("seed", 13))
        self.keep_alive = lc.get("keep_alive", "30m")
        self.timeout = float(lc.get("timeout_s", 900))
        self.retries = int(lc.get("retries", 3))
        self.autostart = bool(lc.get("autostart", True))
        self._seen_up = False                 # o servidor já respondeu alguma vez nesta rodada?
        self._server_error: str | None = None
        self._server_lock = threading.Lock()
        self.think = lc.get("think")
        self.models = cfg.get("models", {})
        self._db_lock = threading.Lock()
        self._local = threading.local()
        self.totals: dict = defaultdict(lambda: defaultdict(float))
        self._totals_lock = threading.Lock()
        path = cfg["paths"]["cache"]
        if path != ":memory:":
            ensure_dir(os.path.dirname(path) or ".")
        self.db = sqlite3.connect(path, check_same_thread=False, timeout=60)
        with self._db_lock:
            try:
                self.db.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
            self.db.commit()
        # fecha o cache quando o cliente deixa de existir (ou na saída do Python), em vez de deixar a conexão aberta
        self._closer = weakref.finalize(self, self.db.close)
        self._client = None
        self._mock = None
        if self.backend == "mock":
            from .mock import MockBrain

            self._mock = MockBrain(seed=self.seed)
        self._logprobs_supported: bool | None = None

    def close(self) -> None:
        """Fecha o cache em disco e a conexão HTTP. Seguro de chamar mais de uma vez."""
        self._closer()
        if self._client is not None:
            try:
                self._client.close()
            finally:
                self._client = None

    # ------------------------------------------------------------------ util
    @property
    def client(self):
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def resolve(self, role_or_model: str) -> str:
        m = self.models.get(role_or_model, role_or_model)
        if not isinstance(m, str):
            raise LLMError(f"papel {role_or_model!r} não aponta para um modelo (valor: {m!r})")
        return m

    # ------------------------------------------------------------------ contabilidade
    @contextlib.contextmanager
    def track(self):
        """Soma chamadas/tokens do bloco (apenas da thread atual)."""
        u = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0, "tokens": 0,
             "llm_seconds": 0.0, "by_task": defaultdict(int)}
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = self._local.stack = []
        stack.append(u)
        try:
            yield u
        finally:
            stack.remove(u)
            u["by_task"] = dict(u["by_task"])

    def _account(self, task: str, pt: int, ct: int, latency: float, cached: bool) -> None:
        with self._totals_lock:
            t = self.totals[task]
            t["calls"] += 1
            t["cached"] += int(cached)
            t["prompt_tokens"] += pt
            t["completion_tokens"] += ct
            t["llm_seconds"] += latency
        for u in getattr(self._local, "stack", []) or []:
            u["calls"] += 1
            u["cached"] += int(cached)
            u["prompt_tokens"] += pt
            u["completion_tokens"] += ct
            u["tokens"] += pt + ct
            u["llm_seconds"] += latency
            u["by_task"][task] += 1

    def summary(self) -> dict:
        with self._totals_lock:
            return {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in self.totals.items()}

    # ------------------------------------------------------------------ cache
    def _key(self, req: dict) -> str:
        return hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()

    def _get(self, key: str):
        with self._db_lock:
            row = self.db.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def _put(self, key: str, val: dict) -> None:
        with self._db_lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)", (key, json.dumps(val, ensure_ascii=False)))
            self.db.commit()

    # ------------------------------------------------------------------ API principal
    def chat(self, role: str, messages: list[dict], task: str, **kw) -> str:
        return self.chat_full(role, messages, task, **kw)["text"]

    def chat_full(self, role: str, messages: list[dict], task: str, max_tokens: int = 256,
                  temperature: float = 0.0, fmt: dict | str | None = None, stop: list[str] | None = None,
                  template: str | None = None, logprobs: bool = False, use_cache: bool = True,
                  options: dict | None = None, seed: int | None = None, think: bool | None = None) -> dict:
        """Uma chamada. ``role`` é um papel do config (generator, critic, planner...) ou o nome do modelo.

        ``options`` são parâmetros extras de amostragem (top_p, top_k, repeat_penalty), ``seed`` troca a
        semente só desta chamada (para repetir uma pergunta com outro sorteio) e ``think`` liga/desliga o modo
        de raciocínio de modelos como o Qwen3. Os três só entram na chave do cache quando são usados, para que o cache de
        rodadas antigas continue valendo.
        """
        model = self.resolve(role)
        req = {"backend": self.backend, "model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temperature, "fmt": fmt, "stop": stop, "template": template,
               "num_ctx": self.num_ctx, "seed": self.seed, "logprobs": bool(logprobs)}
        if options:
            req["options"] = dict(sorted(options.items()))
        if seed is not None and int(seed) != self.seed:
            req["seed"] = int(seed)
        if think is not None:
            req["think"] = bool(think)
        key = self._key(req)
        if use_cache:
            hit = self._get(key)
            if hit is not None:
                self._account(task, hit["pt"], hit["ct"], hit.get("latency", 0.0), cached=True)
                return {**hit, "cached": True}

        t0 = time.time()
        call_seed = int(req["seed"])
        if self.backend == "mock":
            text = self._mock.respond(task, messages, fmt=fmt, salt=call_seed if call_seed != self.seed else None)
            lp = self._mock.logprobs(text) if logprobs else None
        elif self.backend == "ollama":
            text, lp = self._ollama(model, messages, max_tokens, temperature, fmt, stop, template, logprobs,
                                    options, call_seed, think)
        elif self.backend == "openai":
            text, lp = self._openai(model, messages, max_tokens, temperature, fmt, stop, logprobs, options, call_seed)
        else:
            raise LLMError(f"backend desconhecido: {self.backend}")
        latency = time.time() - t0
        prompt_text = render_chatml(messages) if template == "chatml" else None
        pt = count_tokens(prompt_text) if prompt_text else count_messages(messages)
        ct = count_tokens(text)
        val = {"text": text, "pt": pt, "ct": ct, "latency": round(latency, 4), "logprobs": lp}
        if use_cache:
            self._put(key, val)
        self._account(task, pt, ct, latency, cached=False)
        return {**val, "cached": False}

    # ------------------------------------------------------------------ backends
    def _open_server(self) -> None:
        """Ninguém respondeu em ``base_url`` e o servidor ainda não foi visto nesta rodada: ele não está aberto.

        Abre o Ollama desta máquina (``llm.autostart``); se não der, levanta ``LLMError`` já, com o que fazer.
        Uma queda no meio da rodada (o servidor já tinha respondido) não passa por aqui: segue as tentativas normais.
        """
        from . import server as S

        with self._server_lock:
            if self._seen_up:                              # outra thread já abriu
                return
            if self._server_error is None:
                log = get_logger()
                if S.wait_up(self.base_url, 3):            # estava acabando de subir
                    self._seen_up = True
                    return
                if self.backend == "ollama" and self.autostart and S.is_local_ollama(self.base_url):
                    log.warning(f"o Ollama não está aberto ({self.base_url} não responde); abrindo em segundo plano...")
                    how = S.start_local_ollama(self.base_url)
                    if how:
                        log.info(f"Ollama aberto ({how}); ele continua aberto depois da rodada.")
                        self._seen_up = True
                        return
                if self.backend == "ollama":
                    self._server_error = (f"o Ollama não está respondendo em {self.base_url}. Abra o app do Ollama (menu "
                                          "Iniciar → Ollama) ou rode `ollama serve` em outra janela, e repita o comando.")
                else:
                    self._server_error = (f"nada responde em {self.base_url}. Confira se o servidor está no ar e o "
                                          "llm.base_url do config, e repita o comando.")
            raise LLMError(self._server_error)

    def _post(self, url: str, body: dict, headers: dict | None = None) -> dict:
        import httpx

        last = None
        attempt = 0
        while attempt <= self.retries:
            try:
                r = self.client.post(url, json=body, headers=headers or {})
                self._seen_up = True
                if r.status_code == 404:
                    raise LLMError(f"modelo não encontrado no servidor: {body.get('model')}  →  ollama pull {body.get('model')}")
                if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                    # requisição inválida: não adianta repetir; vira erro desta pergunta (o job continua)
                    raise RuntimeError(f"o servidor recusou a requisição ({r.status_code}): {r.text[:300]}")
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(f"{r.status_code}: {r.text[:300]}", request=r.request, response=r)
                r.raise_for_status()
                return r.json()
            except LLMError:
                raise
            except (httpx.HTTPError, json.JSONDecodeError) as e:  # conexão caiu, timeout, 5xx
                if isinstance(e, httpx.ConnectError) and not self._seen_up:
                    self._open_server()     # servidor fechado: abre o Ollama local ou para já (LLMError)
                    continue                # aberto: repete a chamada, sem gastar uma tentativa
                last = e
                wait = [3, 10, 30, 60][min(attempt, 3)]
                get_logger().warning(f"LLM falhou ({type(e).__name__}: {str(e)[:160]}); tentando de novo em {wait}s")
                time.sleep(wait)
                attempt += 1
        raise LLMError(f"LLM indisponível depois de {self.retries + 1} tentativas: {last}")

    def _ollama(self, model, messages, max_tokens, temperature, fmt, stop, template, logprobs,
                extra: dict | None = None, seed: int | None = None, think: bool | None = None):
        options = {"temperature": temperature, "seed": self.seed if seed is None else seed, "num_ctx": self.num_ctx,
                   "num_predict": max_tokens}
        for k, v in (extra or {}).items():
            if k in ("top_p", "top_k", "repeat_penalty", "min_p", "repeat_last_n", "presence_penalty",
                     "frequency_penalty"):
                options[k] = v
        if stop:
            options["stop"] = list(stop)
        if template == "chatml":
            if stop is None:
                options["stop"] = ["<|im_end|>", "<|endoftext|>"]
            body = {"model": model, "prompt": render_chatml(messages), "raw": True, "stream": False,
                    "keep_alive": self.keep_alive, "options": options}
            endpoint = "/api/generate"
        else:
            body = {"model": model, "messages": messages, "stream": False, "keep_alive": self.keep_alive,
                    "options": options}
            endpoint = "/api/chat"
        if fmt is not None:
            body["format"] = fmt
        if think is not None:
            body["think"] = bool(think)
        elif self.think is not None:
            body["think"] = self.think
        if logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = 0
        j = self._post(self.base_url + endpoint, body)
        text = j.get("response") if template == "chatml" else (j.get("message") or {}).get("content", "")
        lp = None
        if logprobs:
            raw_lp = j.get("logprobs")
            if isinstance(raw_lp, list) and raw_lp:
                lp = [{"token": x.get("token", ""), "logprob": float(x.get("logprob", 0.0))} for x in raw_lp
                      if isinstance(x, dict)]
            self._logprobs_supported = lp is not None
        return text or "", lp

    def _openai(self, model, messages, max_tokens, temperature, fmt, stop, logprobs,
                extra: dict | None = None, seed: int | None = None):
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
                "seed": self.seed if seed is None else seed}
        for k, v in (extra or {}).items():       # top_k / repetition_penalty: aceitos pelo vLLM e llama.cpp
            body[{"repeat_penalty": "repetition_penalty"}.get(k, k)] = v
        if stop:
            body["stop"] = list(stop)
        if isinstance(fmt, dict):
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": fmt}}
        elif fmt == "json":
            body["response_format"] = {"type": "json_object"}
        if logprobs:
            body["logprobs"] = True
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        j = self._post(self.base_url + "/v1/chat/completions", body, headers)
        ch = j["choices"][0]
        lp = None
        if logprobs and ch.get("logprobs") and ch["logprobs"].get("content"):
            lp = [{"token": x.get("token", ""), "logprob": float(x.get("logprob", 0.0))} for x in ch["logprobs"]["content"]]
        return ch["message"].get("content") or "", lp

    # ------------------------------------------------------------------ diagnóstico
    def logprobs_supported(self) -> bool | None:
        return self._logprobs_supported
