"""Laço genérico de avaliação: retomável, com contabilidade por pergunta e (opcional) threads.

Cada método recebe um registro e devolve um dicionário com pelo menos ``pred``. O runner acrescenta
EM/F1, uso do LLM (chamadas, tokens, segundos de LLM) e grava uma linha por pergunta em JSONL.
Se o processo cair, rodar de novo continua de onde parou (e o cache do LLM evita recomputar).
"""
from __future__ import annotations

import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from tqdm import tqdm

from .io import append_jsonl, get_logger, read_jsonl, write_jsonl
from .llm import LLMError
from .text import best_em_f1, contains_answer


def score(rec: dict, pred: str) -> dict:
    em, f1 = best_em_f1(pred, rec.get("answers", []))
    return {"em": em, "f1": f1, "acc": contains_answer(pred, rec.get("answers", []))}


def load_done(path: str) -> dict:
    return {r["id"]: r for r in read_jsonl(path) if not r.get("error")}


def run_records(records: list[dict], fn: Callable[[dict], dict], out_path: str, llm, desc: str,
                workers: int = 1, max_error_rate: float = 0.2) -> list[dict]:
    log = get_logger()
    done = load_done(out_path)
    # regrava sem as linhas com erro (serão tentadas de novo); mantém as linhas de outras amostras do mesmo arquivo
    if os.path.exists(out_path) and len(done) != len(read_jsonl(out_path)):
        write_jsonl(out_path, list(done.values()))
    todo = [r for r in records if r["id"] not in done]
    if not todo:
        return [done[r["id"]] for r in records if r["id"] in done]
    n_err = 0

    def work(rec: dict) -> dict:
        t0 = time.time()
        with llm.track() as u:
            try:
                res = fn(rec) or {}
                err = None
            except LLMError:
                raise
            except Exception as e:  # noqa: BLE001
                res, err = {"pred": ""}, f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}"
        row = {"id": rec["id"], "question": rec["question"], "answers": rec.get("answers", []),
               "type": rec.get("type", ""), "hops": rec.get("hops"), **res}
        row.update(score(rec, row.get("pred", "")))
        row["usage"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in u.items()}
        row["llm_calls"] = u["calls"]
        row["tokens"] = u["tokens"]
        row["llm_seconds"] = round(u["llm_seconds"], 3)
        row["secs"] = round(time.time() - t0, 3)
        if err:
            row["error"] = err
        append_jsonl(out_path, row)
        return row

    bar = tqdm(total=len(todo), desc=desc, dynamic_ncols=True, mininterval=2)
    results = []
    try:
        if workers <= 1:
            for rec in todo:
                row = work(rec)
                results.append(row)
                n_err += bool(row.get("error"))
                bar.update(1)
                if n_err >= 10 and n_err / len(results) > max_error_rate:
                    raise RuntimeError(f"{desc}: {n_err}/{len(results)} perguntas com erro — abortando. "
                                       f"Último erro:\n{row.get('error')}")
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(work, rec) for rec in todo]
                for f in as_completed(futs):
                    row = f.result()
                    results.append(row)
                    n_err += bool(row.get("error"))
                    bar.update(1)
    finally:
        bar.close()
    if n_err:
        log.warning(f"{desc}: {n_err} perguntas com erro (serão tentadas de novo na próxima execução). "
                    f"Veja o campo 'error' em {out_path}")
    rows = {r["id"]: r for r in read_jsonl(out_path)}
    return [rows[r["id"]] for r in records if r["id"] in rows]
