"""Entrada/saída: jsonl, json, diretórios e log (console + arquivo)."""
from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import Iterable

_LOCK = threading.Lock()
_WINDOWS = os.name == "nt"
_TRIES, _WAIT = 60, 0.05            # até ~3 s


def _retry(fn):
    """Repete ``fn`` enquanto o Windows responder "arquivo em uso" (PermissionError).

    No Windows não dá para trocar um arquivo (``os.replace``), nem às vezes para abri-lo, no instante em que outro
    processo o está usando: o painel lendo o ``status.json`` enquanto a rodada o regrava, o antivírus, o indexador
    de busca. O bloqueio dura milissegundos; esperar e tentar de novo resolve. Sem isto, uma rodada de horas
    poderia parar porque alguém olhou o painel na hora errada. Fora do Windows o erro sobe na primeira tentativa.
    """
    for i in range(_TRIES):
        try:
            return fn()
        except PermissionError:
            if not _WINDOWS or i == _TRIES - 1:
                raise
            time.sleep(_WAIT)


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def move_file(src: str, dst: str) -> str:
    """Move um arquivo (mesma lógica de repetição do Windows: o painel pode estar lendo o arquivo nesse instante)."""
    import shutil

    ensure_dir(os.path.dirname(dst) or ".")
    _retry(lambda: shutil.move(src, dst))
    return dst


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with _retry(lambda: open(path, encoding="utf-8")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # linha truncada (processo interrompido no meio da escrita): ignora
                continue
    return rows


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    _retry(lambda: os.replace(tmp, path))


def append_jsonl(path: str, row: dict) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with _LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str, default=None):
    if not os.path.exists(path):
        return default
    with _retry(lambda: open(path, encoding="utf-8")) as f:
        return json.load(f)


def write_json(path: str, obj, indent: int = 1) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    _retry(lambda: os.replace(tmp, path))


def setup_logging(log_file: str | None = None, level=logging.INFO) -> logging.Logger:
    log = logging.getLogger("ragbench")
    if getattr(log, "_configured", False):
        if log_file and not any(getattr(h, "baseFilename", None) == os.path.abspath(log_file) for h in log.handlers):
            ensure_dir(os.path.dirname(log_file) or ".")
            fh = logging.FileHandler(log_file, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            log.addHandler(fh)
        return log
    log.setLevel(level)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    log.addHandler(sh)
    if log_file:
        ensure_dir(os.path.dirname(log_file) or ".")
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(fh)
    log.propagate = False
    log._configured = True  # type: ignore[attr-defined]
    return log


def get_logger() -> logging.Logger:
    return setup_logging()
