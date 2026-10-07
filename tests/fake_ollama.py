"""Servidor falso com a API do Ollama (/api/chat, /api/generate, /api/embed, /api/tags, /api/ps, /api/version).

Serve para testar o caminho HTTP real do cliente (corpo das requisições, parse das respostas, logprobs,
ChatML cru) sem GPU. As respostas vêm do MockBrain, escolhendo a "tarefa" pelo conteúdo do prompt.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ragbench.core.mock import MockBrain
from ragbench.core.retrieval import HashEncoder

BRAIN = MockBrain()
MODELS = ["qwen2.5:3b", "hf.co/mradermacher/RAG-Critic-3B-GGUF:Q4_K_M", "llama3.2:3b"]
EMBED_MODELS = ["hash"]          # "modelo de embedding" de teste: o mesmo HashEncoder do pacote
EMBED_NOISE = [0.0]              # > 0 simula um modelo do Ollama que NÃO coincide com o da CPU
LOG: list[dict] = []


def _task(text: str, fmt) -> str:
    t = text
    if "critical system designed to provide useful error type tags" in t:
        return "rc_critic_zeroshot" if "tag1 corresponds to tag2" in t else "rc_critic"
    if "optimizing a Retrieval-Augmented Generation process" in t:
        return "rc_plan"
    if isinstance(fmt, dict):
        props = fmt.get("properties", {})
        if "labels" in props:
            return "crag_eval"
        if "gaps" in props:
            return "s2g_judge"
        if "subquestions" in props:
            return "par2_decompose"
        if "docs_sufficient" in props:
            return "metarag_eval"
        if "label" in props:
            return "adaptive_classify"
        return "dk_suff_judge"
    if "Missing information:" in t:
        return "dk_gap"
    if "New query:" in t:
        return "dk_rewrite"
    if "Snippets" in t or "fact table" in t or "Output:" in t:
        return "dk_compress"
    if "Evidence:" in t and "Answer:" in t:
        return "dk_answer"
    if "**Question**" in t:
        return "rc_generate"
    return "default"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silencioso
        pass

    def _send(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/api/version":
            return self._send({"version": "0.12.99-fake"})
        if self.path == "/api/tags":
            return self._send({"models": [{"name": m, "model": m, "size": 2 * 2 ** 30} for m in MODELS + EMBED_MODELS]})
        if self.path == "/api/ps":
            return self._send({"models": [{"name": m, "model": m, "size": 2 * 2 ** 30, "size_vram": 2 * 2 ** 30} for m in MODELS]})
        return self._send({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        LOG.append({"path": self.path, "body": body})
        if self.path == "/api/embed":
            if body.get("model") not in EMBED_MODELS:
                return self._send({"error": f"model '{body.get('model')}' not found"}, 404)
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            emb = HashEncoder().encode(texts)
            if EMBED_NOISE[0]:
                import numpy as np

                emb = emb + EMBED_NOISE[0] * np.random.default_rng(0).standard_normal(emb.shape).astype("float32")
            return self._send({"model": body["model"], "embeddings": emb.tolist()})
        if body.get("model") not in MODELS + EMBED_MODELS:
            return self._send({"error": f"model '{body.get('model')}' not found"}, 404)
        if self.path == "/api/generate" and "prompt" not in body:          # keep_alive=0: descarrega o modelo
            return self._send({"model": body["model"], "done": True, "done_reason": "unload"})
        if self.path == "/api/chat":
            msgs = body["messages"]
            text = "\n".join(m["content"] for m in msgs)
            out = BRAIN.respond(_task(text, body.get("format")), msgs, fmt=body.get("format"))
            resp = {"model": body["model"], "message": {"role": "assistant", "content": out}, "done": True,
                    "prompt_eval_count": 10, "prompt_eval_duration": 10_000_000, "eval_count": 5, "eval_duration": 50_000_000}
        elif self.path == "/api/generate":
            prompt = body["prompt"]
            out = BRAIN.respond(_task(prompt, body.get("format")), [{"role": "user", "content": prompt}])
            resp = {"model": body["model"], "response": out, "done": True, "prompt_eval_count": 10,
                    "prompt_eval_duration": 10_000_000, "eval_count": 5, "eval_duration": 50_000_000}
        else:
            return self._send({"error": "not found"}, 404)
        if body.get("logprobs"):
            resp["logprobs"] = [{"token": t, "logprob": -0.05 if i % 4 else -2.0} for i, t in enumerate(out.split(" "))]
        return self._send(resp)


def start(port: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"
