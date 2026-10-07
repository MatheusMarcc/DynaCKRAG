"""Dados de treino do crítico: SFT (Prompt D → JSON) e pares DPO coarse-to-fine.

Fontes aceitas:
  * ``official`` — o dataset público ``dongguanting/RAG-Error-Critic-100K`` (Hugging Face), no formato
    LLaMA-Factory {instruction, input, output};
  * ``local`` — o D_Error^QA construído por ``ragcritic errorsystem`` (+ as respostas corretas).
"""
from __future__ import annotations

import os
import random

from ...core.io import ensure_dir, get_logger, read_jsonl, write_jsonl
from ...core.llm import QWEN_DEFAULT_SYSTEM
from ...core.jsonutil import parse_obj
from .. import prompts as P

OFFICIAL_REPO = "dongguanting/RAG-Error-Critic-100K"


def _label(output: str) -> str:
    obj = parse_obj(output, {}) or {}
    j = str(obj.get("Judgement", "") if isinstance(obj, dict) else "").lower()
    return "Correct" if j.startswith("correct") else "Error"


def load_official(limit: int | None = None, seed: int = 13) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(OFFICIAL_REPO, split="train", streaming=True).shuffle(seed=seed, buffer_size=10_000)
    rows = []
    for ex in ds:
        ins = (ex.get("instruction") or "") + (("\n" + ex["input"]) if ex.get("input") else "")
        out = ex.get("output") or ""
        if ins and out:
            rows.append({"instruction": ins, "output": out if isinstance(out, str) else str(out), "label": _label(str(out))})
        if limit and len(rows) >= limit:
            break
    return rows


def load_local(results_dir: str, indent: int = 20) -> list[dict]:
    es = os.path.join(results_dir, "ragcritic", "errorsystem")
    rows = []
    for x in read_jsonl(os.path.join(es, "D_error_qa.jsonl")):
        out = str({"Judgement": "Error", "Error_analysis": x.get("analysis", ""), "tag1": x["tag1"], "tag2": x["tag2"],
                   "tag3": x["tag3"]})
        rows.append({"instruction": P.render_critic(x["question"], x["contents"][:5], x["prediction"], indent, "critic"),
                     "output": out, "label": "Error"})
    for x in read_jsonl(os.path.join(es, "D_correct.jsonl")):
        rows.append({"instruction": P.render_critic(x["question"], x["contents"][:5], x["prediction"], indent, "critic"),
                     "output": "{'Judgement': 'Correct'}", "label": "Correct"})
    return rows


def neutralize_indent(rows: list[dict], indent: int = 20) -> list[dict]:
    """Re-renderiza todos os prompts com o MESMO recuo (remove o atalho de formatação do treino)."""
    out = []
    for r in rows:
        try:
            p = P.parse_critic_instruction(r["instruction"])
            ins = P.render_critic(p["question"], p["passages"], p["prediction"], indent, p["kind"])
        except (ValueError, AttributeError):
            ins = r["instruction"]
        out.append({**r, "instruction": ins})
    return out


def build_sft(rows: list[dict], seed: int = 13) -> list[dict]:
    """"We first randomly select several correct samples equal to that in D_Error^QA" — balanceia 1:1."""
    rng = random.Random(seed)
    err = [r for r in rows if r["label"] == "Error"]
    cor = [r for r in rows if r["label"] == "Correct"]
    rng.shuffle(cor)
    data = err + cor[: len(err)]
    rng.shuffle(data)
    return [{"prompt": [{"role": "system", "content": QWEN_DEFAULT_SYSTEM}, {"role": "user", "content": r["instruction"]}],
             "completion": [{"role": "assistant", "content": r["output"]}], "label": r["label"]} for r in data]


def build_dpo(rows: list[dict], k: int = 1, seed: int = 13) -> list[dict]:
    """Pares (y+, y−): negativos do conjunto de CORRETOS (grosso) e do conjunto de ERROS (fino)."""
    rng = random.Random(seed)
    err = [r for r in rows if r["label"] == "Error"]
    correct_out = "{'Judgement': 'Correct'}"
    pairs = []
    for r in rows:
        prompt = [{"role": "system", "content": QWEN_DEFAULT_SYSTEM}, {"role": "user", "content": r["instruction"]}]
        negs = []
        if r["label"] == "Error":
            negs.append(correct_out)                                                    # grosso: "está correto"
            others = [e["output"] for e in rng.sample(err, min(len(err), 4 * k)) if e["output"] != r["output"]]
            negs += others[:k]                                                          # fino: outras tags de erro
        else:
            negs += [e["output"] for e in rng.sample(err, min(len(err), k))]            # grosso: "tem erro"
        for n in negs:
            pairs.append({"prompt": prompt, "chosen": [{"role": "assistant", "content": r["output"]}],
                          "rejected": [{"role": "assistant", "content": n}],
                          "kind": "coarse" if (n == correct_out or r["label"] == "Correct") else "fine"})
    rng.shuffle(pairs)
    return pairs


def build(cfg: dict, source: str = "official", limit: int | None = 20000, k: int = 1, neutral_indent: bool = True) -> dict:
    log = get_logger()
    seed = int(cfg.get("seed", 13))
    rows = load_official(limit, seed) if source == "official" else load_local(cfg["paths"]["results"])
    if neutral_indent:
        rows = neutralize_indent(rows)
    # no perfil de teste os dados sintéticos ficam com os outros dados de teste, fora da pasta dos dados de verdade
    base = cfg["paths"]["data"] if cfg["data"]["scale"] == "smoke" else cfg["paths"]["data_root"]
    out_dir = ensure_dir(os.path.join(base, "critic_training"))
    sft = build_sft(rows, seed)
    dpo = build_dpo(rows, k, seed)
    write_jsonl(os.path.join(out_dir, f"sft_{source}.jsonl"), sft)
    write_jsonl(os.path.join(out_dir, f"dpo_{source}.jsonl"), dpo)
    info = {"source": source, "n_rows": len(rows), "n_sft": len(sft), "n_dpo": len(dpo), "dir": out_dir,
            "neutral_indent": neutral_indent}
    log.info(f"dados de treino do crítico: {info}")
    return info
