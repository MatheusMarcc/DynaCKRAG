"""O servidor do LLM está aberto? Usado quando a primeira chamada nova de uma rodada encontra a porta fechada.

Uma rodada que sai inteira do cache não chega aqui e continua funcionando sem servidor nenhum. Quando a primeira
chamada que precisa do modelo não acha ninguém no endereço, a causa quase sempre é o Ollama não estar aberto:
abrimos o Ollama desta máquina (``ollama serve`` em segundo plano, o mesmo que o ``run_all.ps1`` faz) e, se não
der, a rodada para na hora dizendo o que fazer — em vez de repetir a chamada por dois minutos.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from urllib.parse import urlparse

OLLAMA_PORT = 11434
LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")
_STARTED: list = []          # servidores iniciados pelo pacote; guardar a referência evita o aviso de processo ainda rodando


def is_up(base_url: str, timeout: float = 3.0) -> bool:
    """Alguém responde HTTP neste endereço? Qualquer resposta serve (um 404 também é um servidor no ar)."""
    import httpx

    try:
        httpx.get(base_url.rstrip("/") + "/api/version", timeout=timeout)
        return True
    except (httpx.HTTPError, OSError):
        return False


def wait_up(base_url: str, seconds: float, step: float = 1.0) -> bool:
    """Espera até ``seconds`` o servidor responder."""
    end = time.monotonic() + seconds
    while True:
        if is_up(base_url):
            return True
        if time.monotonic() >= end:
            return False
        time.sleep(step)


def is_local_ollama(base_url: str) -> bool:
    """É o endereço padrão do Ollama desta máquina? (o único servidor que sabemos abrir)"""
    try:
        u = urlparse(base_url if "//" in base_url else "//" + base_url)
        return (u.hostname or "").lower() in LOCAL_HOSTS and u.port == OLLAMA_PORT
    except ValueError:
        return False


def start_local_ollama(base_url: str, wait_s: float = 45.0) -> str | None:
    """Abre o Ollama desta máquina em segundo plano e espera responder.

    Devolve como abriu (texto para o log) ou None se não conseguiu: o comando ``ollama`` não está no PATH,
    não deu para iniciar o processo ou ele não respondeu em ``wait_s`` segundos.
    """
    exe = shutil.which("ollama")
    if not exe:
        return None
    kw: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        # sem janela e fora do grupo do terminal: o servidor continua aberto depois que a rodada termina (ou leva Ctrl+C)
        kw["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kw["start_new_session"] = True
    try:
        _STARTED.append(subprocess.Popen([exe, "serve"], **kw))
    except OSError:
        return None
    return "ollama serve" if wait_up(base_url, wait_s) else None
