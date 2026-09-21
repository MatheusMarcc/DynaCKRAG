"""Crítico RAG-Critic-3B: julga a resposta e devolve tags de erro em 3 níveis.

Detalhes que importam para o modelo se comportar como no artigo:
  * o prompt inteiro vai na mensagem de SISTEMA e a mensagem de usuário fica vazia
    (é assim que o repositório oficial chama o modelo);
  * top-5 passagens no formato "Título"\\ntexto;
  * a chave real da análise é "Error_analysis" (o PDF escreve "Error analysis").
"""
from __future__ import annotations

import json
import os

import json_repair
from tqdm import tqdm

from .prompts import critic_prompt
from .utils import passage_str, read_jsonl, write_jsonl


def parse_critic(text: str) -> dict:
    j = json_repair.loads(text or "")
    if isinstance(j, list):
        j = next((x for x in j if isinstance(x, dict)), {})
    if not isinstance(j, dict):
        j = {}
    verdict = str(j.get("Judgement", j.get("judgement", ""))).lower()
    as_list = lambda v: v if isinstance(v, list) else ([v] if v else [])  # noqa: E731
    return {
        "judgement": "error" if ("error" in verdict or "incorrect" in verdict) else ("correct" if verdict else "unknown"),
        "analysis": j.get("Error_analysis") or j.get("Error analysis") or j.get("error_analysis") or "",
        "tag1": as_list(j.get("tag1")), "tag2": as_list(j.get("tag2")), "tag3": as_list(j.get("tag3")),
    }


def critic_docs(state_docs: list[dict], k: int) -> list[dict]:
    """O crítico viu top-5 no treino: mandamos os k docs de maior escore de retrieval."""
    return sorted(state_docs, key=lambda d: -d.get("score", 0.0))[:k]


def judge(llm, question: str, docs: list[dict], prediction: str) -> dict:
    prompt = critic_prompt(question, [passage_str(d) for d in docs], prediction)
    raw = llm.chat("critic", [{"role": "system", "content": prompt}, {"role": "user", "content": ""}],
                   task="critic", max_tokens=512)
    out = parse_critic(raw)
    out["raw"] = raw
    return out


def run_critic(cfg: dict, llm, run_name: str) -> str:
    run_dir = os.path.join(cfg["paths"]["runs"], run_name)
    rows = read_jsonl(os.path.join(run_dir, "preds.jsonl"))
    k = cfg["repair"]["critic_top_k"]
    out = []
    for r in tqdm(rows, desc="crítico"):
        with llm.track() as u:
            c = judge(llm, r["question"], critic_docs(r["state"]["docs"], k), r["raw"] or r["pred"])
        out.append({"qid": r["qid"], "em": r["em"], **c, "usage": dict(u)})
    path = os.path.join(run_dir, "critic.jsonl")
    write_jsonl(path, out)
    # concordância crítico × métrica (EM) — diagnóstico, usa o gold só para avaliar o crítico
    agree = sum((o["judgement"] == "correct") == (o["em"] == 1.0) for o in out) / max(1, len(out))
    flagged = sum(o["judgement"] == "error" for o in out)
    print(f"crítico marcou {flagged}/{len(out)} como erro | concordância com EM: {agree:.1%}")
    return path


def critic_bench(llm, bench_file: str, n: int | None = None) -> dict:
    """Valida o crítico no RAG-Error Bench do repositório oficial (critic_model_test.json)."""
    import random

    with open(bench_file, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    random.Random(13).shuffle(items)          # o arquivo vem ordenado (corretos e erros em blocos)
    items = items[:n]
    stats = {"correct": [0, 0], "error": [0, 0]}
    tag_f1 = []
    for it in tqdm(items, desc="RAG-Error Bench"):
        gold = it["output"] if isinstance(it["output"], dict) else json_repair.loads(str(it["output"]))
        g = "error" if "error" in str(gold.get("Judgement", "")).lower() else "correct"
        raw = llm.chat("critic", [{"role": "system", "content": it["instruction"]}, {"role": "user", "content": ""}],
                       task="critic_bench", max_tokens=512)
        p = parse_critic(raw)
        stats[g][0] += int(p["judgement"] == g)
        stats[g][1] += 1
        if g == "error" and p["judgement"] == "error":
            gs, ps = set(gold.get("tag1", [])), set(p["tag1"])
            if gs or ps:
                tp = len(gs & ps)
                tag_f1.append(0.0 if tp == 0 else 2 * tp / (len(gs) + len(ps)))
    res = {k: (v[0] / v[1] if v[1] else None) for k, v in stats.items()}
    vals = [x for x in res.values() if x is not None]
    res["avg_identification"] = sum(vals) / len(vals) if vals else None
    res["tag1_f1"] = sum(tag_f1) / len(tag_f1) if tag_f1 else None
    print(json.dumps(res, indent=2))
    return res
