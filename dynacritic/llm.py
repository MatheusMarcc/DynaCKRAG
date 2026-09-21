"""Cliente de LLM com cache em disco e contabilidade de tokens.

Backends:
  * ollama  — API nativa /api/chat (permite num_ctx, seed, JSON schema e keep_alive por chamada)
  * openai  — /v1/chat/completions (vLLM, llama.cpp server, APIs pagas): troca só no config
  * mock    — respostas heurísticas determinísticas, para testar o pipeline sem GPU

O cache (SQLite) é essencial num notebook: com temperatura 0, rodar de novo um experimento
não custa nada — só as chamadas novas vão ao modelo.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sqlite3
import threading
from collections import defaultdict

import httpx


class LLM:
    def __init__(self, cfg: dict):
        lc = cfg["llm"]
        self.backend = lc["backend"]
        self.base_url = lc["base_url"].rstrip("/")
        self.api_key = lc.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        self.num_ctx = lc.get("num_ctx", 8192)
        self.seed = lc.get("seed", 13)
        self.keep_alive = lc.get("keep_alive", "30m")
        self.think = lc.get("think")
        self.timeout = lc.get("timeout_s", 600)
        self.models = cfg["models"]
        self._lock = threading.Lock()
        path = lc.get("cache_path") or ":memory:"
        if path != ":memory:":
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self.totals = defaultdict(lambda: defaultdict(int))  # task -> {calls, prompt, completion, cached}
        self._trackers: list[dict] = []
        self._client = httpx.Client(timeout=self.timeout)
        if self.backend == "mock":
            from .mock import MockBrain

            self._mock = MockBrain()

    # ------------------------------------------------------------ contabilidade
    @contextlib.contextmanager
    def track(self):
        """with llm.track() as u: ...  -> u = {calls, prompt_tokens, completion_tokens, by_task}"""
        u = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "by_task": defaultdict(int)}
        self._trackers.append(u)
        try:
            yield u
        finally:
            self._trackers.remove(u)
            u["by_task"] = dict(u["by_task"])

    def _account(self, task: str, usage: dict, cached: bool):
        t = self.totals[task]
        t["calls"] += 1
        t["prompt"] += usage.get("prompt", 0)
        t["completion"] += usage.get("completion", 0)
        t["cached"] += int(cached)
        for u in self._trackers:
            u["calls"] += 1
            u["prompt_tokens"] += usage.get("prompt", 0)
            u["completion_tokens"] += usage.get("completion", 0)
            u["by_task"][task] += 1

    # ------------------------------------------------------------ chamada principal
    def chat(self, role: str, messages: list[dict], task: str, max_tokens: int = 256,
             fmt: dict | str | None = None, temperature: float = 0.0) -> str:
        """role: 'generator' | 'critic' | 'planner' (resolvido em config.models)."""
        model = self.models[role]
        req = {"backend": self.backend, "model": model, "messages": messages, "max_tokens": max_tokens,
               "fmt": fmt, "temperature": temperature, "num_ctx": self.num_ctx, "seed": self.seed}
        key = hashlib.sha256(json.dumps(req, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            row = self.db.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
        if row:
            v = json.loads(row[0])
            self._account(task, v["usage"], cached=True)
            return v["text"]

        if self.backend == "ollama":
            text, usage = self._ollama(model, messages, max_tokens, fmt, temperature)
        elif self.backend == "openai":
            text, usage = self._openai(model, messages, max_tokens, fmt, temperature)
        elif self.backend == "mock":
            text = self._mock.respond(task, messages)
            usage = {"prompt": sum(len(m["content"]) for m in messages) // 4, "completion": len(text) // 4}
        else:
            raise ValueError(f"backend desconhecido: {self.backend}")

        with self._lock:
            self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?)",
                            (key, json.dumps({"text": text, "usage": usage}, ensure_ascii=False)))
            self.db.commit()
        self._account(task, usage, cached=False)
        return text

    # ------------------------------------------------------------ backends
    def _ollama(self, model, messages, max_tokens, fmt, temperature):
        body = {
            "model": model, "messages": messages, "stream": False, "keep_alive": self.keep_alive,
            "options": {"temperature": temperature, "seed": self.seed, "num_ctx": self.num_ctx,
                        "num_predict": max_tokens},
        }
        if fmt is not None:
            body["format"] = fmt  # "json" ou um JSON schema (structured outputs)
        if self.think is not None:
            body["think"] = self.think
        r = self._client.post(f"{self.base_url}/api/chat", json=body)
        r.raise_for_status()
        j = r.json()
        return j["message"]["content"], {"prompt": j.get("prompt_eval_count", 0), "completion": j.get("eval_count", 0)}

    def _openai(self, model, messages, max_tokens, fmt, temperature):
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "seed": self.seed}
        if isinstance(fmt, dict):
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "schema": fmt}}
        elif fmt == "json":
            body["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = self._client.post(f"{self.base_url}/v1/chat/completions", json=body, headers=headers)
        r.raise_for_status()
        j = r.json()
        u = j.get("usage", {})
        return j["choices"][0]["message"]["content"], {"prompt": u.get("prompt_tokens", 0),
                                                       "completion": u.get("completion_tokens", 0)}

    def summary(self) -> dict:
        return {k: dict(v) for k, v in self.totals.items()}
