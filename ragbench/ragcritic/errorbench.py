"""RAG-Error Bench (Tabelas 5 e 7, Figs. 5 e 7): identificação e classificação fina de erros.

* Dados: ``rag_error_bench/test_data`` do repositório oficial (1.900 exemplos: 950 corretos + 950 com
  erro, 50 por categoria de nível 2). ``critic_model_test.json`` tem o Prompt D (para o RAG-Critic-3B);
  ``baseline_test.json`` tem o Prompt E, com as listas de tags (para LLMs zero-shot).
* Métricas: porte fiel de ``rag_error_bench/caculate_acc.py`` (inclusive as peculiaridades).
* **Auditoria de formato** (não consta no artigo): nos arquivos oficiais o recuo das linhas separa as
  classes perfeitamente (Correct → 20 espaços, Error → 16). Avaliamos o crítico também com o recuo
  TROCADO. Se a acurácia cair muito, o crítico aprendeu o atalho de formatação, não o conteúdo.
"""
from __future__ import annotations

import json
import os
import random
from collections import defaultdict

from ..core.io import ensure_dir, get_logger, read_json, read_jsonl, write_json
from ..core.jsonutil import parse_obj
from ..core.retrieval import slug
from ..core.runner import run_records
from . import prompts as P
from .critic import Critic, parse_critic_output
from .taxonomy import TIER1, TIER2

RAW_BASE = "https://raw.githubusercontent.com/dongguanting/RAG-Critic/main/rag_error_bench/test_data/"
FILES = {"critic": "critic_model_test.json", "baseline": "baseline_test.json"}


# ============================================================================ dados
def bench_dir(cfg: dict) -> str:
    base = cfg["paths"]["data"] if cfg.get("profile") == "smoke" else cfg["paths"]["data_root"]
    return ensure_dir(os.path.join(base, "rag_error_bench"))


def make_synthetic_bench(cfg: dict, n: int = 24) -> dict:
    """Benchmark sintético (perfil smoke, offline) com o MESMO artefato do oficial (Correct→20, Error→16)."""
    from .taxonomy import TIER2_PARENT

    rng = random.Random(13)
    out = {}
    t2 = list(TIER2_PARENT)
    for kind, fn in FILES.items():
        path = os.path.join(bench_dir(cfg), fn.replace(".json", ".jsonl"))
        rows = []
        for i in range(n):
            err = i % 2 == 1
            q = f"Who founded the fictional company number {i}?"
            passages = [f'"Company {i}"\nCompany {i} was founded by Person {i} in {1900 + i}.'] + \
                       [f'"Other {j}"\nOther {j} is unrelated.' for j in range(4)]
            pred = f"So the final answer is: {'Person ' + str(i + 7) if err else 'Person ' + str(i)}"
            ins = P.render_critic(q, passages, pred, 16 if err else 20, kind)
            if err:
                tag2 = rng.choice(t2)
                gold = {"Judgement": "Error", "Error_analysis": "wrong person", "tag1": [TIER2_PARENT[tag2]],
                        "tag2": [tag2], "tag3": ["Entity Error"]}
            else:
                gold = {"Judgement": "Correct"}
            rows.append({"instruction": ins, "input": "", "output": str(gold), "history": []})
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        out[kind] = path
    return out


def ensure_files(cfg: dict) -> dict:
    """Baixa os dois arquivos do benchmark (≈31 MB) do GitHub, se ainda não existirem."""
    log = get_logger()
    if cfg.get("profile") == "smoke":
        paths = {k: os.path.join(bench_dir(cfg), fn.replace(".json", ".jsonl")) for k, fn in FILES.items()}
        return paths if all(os.path.exists(p) for p in paths.values()) else make_synthetic_bench(cfg)
    out = {}
    for kind, fn in FILES.items():
        path = os.path.join(bench_dir(cfg), fn.replace(".json", ".jsonl"))
        if not os.path.exists(path):
            import httpx

            url = RAW_BASE + fn
            log.info(f"baixando {url}")
            r = httpx.get(url, timeout=300, follow_redirects=True)
            r.raise_for_status()
            with open(path + ".tmp", "wb") as f:
                f.write(r.content)
            os.replace(path + ".tmp", path)
        out[kind] = path
    return out


def load_items(path: str) -> list[dict]:
    items = []
    for i, d in enumerate(read_jsonl(path)):
        gold = parse_obj(d["output"], {}) or {}
        items.append({"id": f"b{i:04d}", "instruction": d["instruction"], "gold": gold,
                      "label": "Correct" if str(gold.get("Judgement", "")).lower().startswith("correct") else "Error"})
    return items


def sample(items: list[dict], n_correct: int | None, n_error: int | None, seed: int = 13) -> list[dict]:
    """Amostra estratificada: corretos ao acaso; erros em rodízio pelas categorias de nível 2."""
    rng = random.Random(seed)
    corr = [x for x in items if x["label"] == "Correct"]
    err = [x for x in items if x["label"] == "Error"]
    rng.shuffle(corr)
    if n_correct is not None:
        corr = corr[:n_correct]
    if n_error is not None and n_error < len(err):
        buckets = defaultdict(list)
        for x in err:
            buckets[(x["gold"].get("tag2") or ["?"])[0]].append(x)
        for b in buckets.values():
            rng.shuffle(b)
        keys = sorted(buckets)
        picked = []
        while len(picked) < n_error and any(buckets.values()):
            for k in keys:
                if buckets[k] and len(picked) < n_error:
                    picked.append(buckets[k].pop())
        err = picked
    out = corr + err
    rng.shuffle(out)
    return out


def swap_indent(instruction: str) -> str:
    """Re-renderiza o mesmo exemplo com o outro recuo (20 ↔ 16), sem mudar nenhuma palavra."""
    p = P.parse_critic_instruction(instruction)
    other = 16 if p["indent"] == 20 else 20
    return P.render_critic(p["question"], p["passages"], p["prediction"], other, p["kind"])


# ============================================================================ métricas oficiais (caculate_acc.py)
def official_metrics(golds: list[dict], outputs: list[dict]) -> dict:
    total = {t: {"correct": 0, "total": 0, "predicted_total": 0} for t in ("tag1", "tag2")}
    cats = {"tag1": {c: {"correct": 0, "total": 0, "predicted_total": 0} for c in TIER1},
            "tag2": {c: {"correct": 0, "total": 0, "predicted_total": 0} for c in TIER2}}
    jud = {"overall": [0, 0], "tag1": [0, 0], "tag2": [0, 0], "correct_case": [0, 0], "error_case": [0, 0]}
    for golden, output in zip(golds, outputs):
        g_j, o_j = golden.get("Judgement"), output.get("Judgement")
        jud["overall"][1] += 1
        jud["overall"][0] += int(g_j == o_j)
        if g_j == "Correct":
            jud["correct_case"][1] += 1
            jud["correct_case"][0] += int(o_j == "Correct")
        else:
            jud["error_case"][1] += 1
            jud["error_case"][0] += int(o_j == "Error")
        for t in ("tag1", "tag2"):
            if t in golden:
                jud[t][1] += 1
                if g_j == "Error" and o_j == "Error" and t in output:
                    jud[t][0] += 1
        if g_j == o_j and g_j == "Correct":
            continue  # o script oficial pula os corretos acertados
        for t in ("tag1", "tag2"):
            valid = TIER1 if t == "tag1" else TIER2
            gt = {x for x in golden.get(t, []) if x in valid}
            ot = {x for x in output.get(t, []) if x in valid}
            m = gt & ot
            total[t]["total"] += len(gt)
            total[t]["predicted_total"] += len(ot)
            total[t]["correct"] += len(m)
            for c in valid:
                cats[t][c]["total"] += c in gt
                cats[t][c]["predicted_total"] += c in ot
                cats[t][c]["correct"] += c in m

    def f1(c, tot, pred):
        if tot == 0 or pred == 0:
            return 0.0
        p, r = c / pred, c / tot
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    ov = {t: {"accuracy": (v["correct"] / v["total"]) if v["total"] else 0.0,
              "f1": f1(v["correct"], v["total"], v["predicted_total"])} for t, v in total.items()}
    ratio = {k: (a / b if b else 0.0) for k, (a, b) in jud.items()}
    return {
        "identification": {"correct": ratio["correct_case"], "error": ratio["error_case"], "avg": ratio["overall"]},
        "classification": {"tag1_f1": ov["tag1"]["f1"], "tag2_f1": ov["tag2"]["f1"],
                           "avg_f1": (ov["tag1"]["f1"] + ov["tag2"]["f1"]) / 2,
                           "tag1_acc": ov["tag1"]["accuracy"], "tag2_acc": ov["tag2"]["accuracy"]},
        "official": {"overall": {"accuracy": (ov["tag1"]["accuracy"] + ov["tag2"]["accuracy"]) / 2,
                                 "f1": (ov["tag1"]["f1"] + ov["tag2"]["f1"]) / 2,
                                 "judgement_accuracy": ratio["overall"],
                                 "correct_judgement_accuracy": ratio["correct_case"],
                                 "tag1": {**ov["tag1"], "judgement_accuracy": ratio["tag1"]},
                                 "tag2": {**ov["tag2"], "judgement_accuracy": ratio["tag2"]}},
                     "category_metrics": {t: {c: {"accuracy": (v["correct"] / v["total"]) if v["total"] else 0.0,
                                                  "f1": f1(v["correct"], v["total"], v["predicted_total"])}
                                              for c, v in cats[t].items()} for t in cats}},
        "n": len(golds),
    }


# ============================================================================ execução
def _out_dir(cfg: dict) -> str:
    return ensure_dir(os.path.join(cfg["paths"]["results"], "ragcritic", "errorbench"))


def run_condition(cfg: dict, llm, critic: Critic, items: list[dict], condition: str, tag: str) -> dict:
    """condition: 'original' (prompt do arquivo) | 'swapped' (recuo trocado)."""
    path = os.path.join(_out_dir(cfg), f"{tag}__{condition}.jsonl")
    recs = [{"id": x["id"], "question": x["id"], "answers": []} for x in items]
    by_id = {x["id"]: x for x in items}

    def fn(rec):
        x = by_id[rec["id"]]
        prompt = x["instruction"] if condition == "original" else swap_indent(x["instruction"])
        out = critic.run_prompt(prompt, task="rc_critic_bench" if critic.mode == "finetuned" else "rc_critic_zeroshot_bench")
        return {"label": x["label"], "judgement": out["judgement"], "tag1": out["tag1"], "tag2": out["tag2"],
                "tag3": out["tag3"], "parse_ok": out["parse_ok"], "raw": out["raw"],
                "indent": P.parse_critic_instruction(prompt)["indent"]}

    rows = run_records(recs, fn, path, llm, desc=f"[RAG-Error Bench] {tag} · {condition}",
                       workers=int(cfg["llm"].get("workers", 1)))
    golds = [by_id[r["id"]]["gold"] for r in rows]
    outs = [{"Judgement": r["judgement"], "tag1": r["tag1"], "tag2": r["tag2"]} for r in rows]
    m = official_metrics(golds, outs)
    m.update(model=tag, condition=condition, parse_fail=sum(not r["parse_ok"] for r in rows),
             unknown=sum(r["judgement"] == "Unknown" for r in rows))
    write_json(os.path.join(_out_dir(cfg), f"{tag}__{condition}.metrics.json"), m)
    return m


def derived_uniform_indent(cfg: dict, tag: str) -> dict:
    """Combina 'original' e 'swapped' para simular TODOS os prompts com o mesmo recuo (20 ou 16)."""
    orig = {r["id"]: r for r in read_jsonl(os.path.join(_out_dir(cfg), f"{tag}__original.jsonl"))}
    swap = {r["id"]: r for r in read_jsonl(os.path.join(_out_dir(cfg), f"{tag}__swapped.jsonl"))}
    out = {}
    for target in (20, 16):
        rows = []
        for i, o in orig.items():
            r = o if o.get("indent") == target else swap.get(i)
            if r is not None:
                rows.append(r)
        if rows:
            golds = [{"Judgement": r["label"]} for r in rows]
            outs = [{"Judgement": r["judgement"]} for r in rows]
            m = official_metrics(golds, outs)
            out[f"all_indent{target}"] = m["identification"]
    write_json(os.path.join(_out_dir(cfg), f"{tag}__uniform_indent.metrics.json"), out)
    return out


def run_errorbench(cfg: dict, llm, which: str = "finetuned", model: str | None = None) -> dict:
    """which='finetuned' → RAG-Critic-3B (Prompt D, com auditoria); which='zeroshot' → ``model`` com Prompt E."""
    eb = cfg["ragcritic"]["errorbench"]
    files = ensure_files(cfg)
    seed = int(cfg.get("seed", 13))
    if which == "finetuned":
        items = sample(load_items(files["critic"]), eb.get("n_correct"), eb.get("n_error"), seed)
        critic = Critic(llm, "finetuned", model="critic", max_tokens=int(cfg["ragcritic"]["critic_max_tokens"]))
        tag = "rag-critic-3b"
        res = {"original": run_condition(cfg, llm, critic, items, "original", tag)}
        if eb.get("format_audit", True):
            res["swapped"] = run_condition(cfg, llm, critic, items, "swapped", tag)
            res["uniform"] = derived_uniform_indent(cfg, tag)
        return res
    items = sample(load_items(files["baseline"]), eb.get("n_correct"), eb.get("n_error"), seed)
    critic = Critic(llm, "zeroshot", model=model, max_tokens=int(cfg["ragcritic"]["critic_max_tokens"]))
    tag = "zeroshot-" + slug(model)
    return {"original": run_condition(cfg, llm, critic, items, "original", tag)}


def format_artifact_stats(cfg: dict) -> dict:
    """Quantifica o artefato: quantos exemplos de cada classe usam cada recuo (em cada arquivo)."""
    files = ensure_files(cfg)
    out = {}
    for kind, path in files.items():
        cnt = defaultdict(int)
        for x in load_items(path):
            cnt[f"{x['label']}@{P.parse_critic_instruction(x['instruction'])['indent']}"] += 1
        out[kind] = dict(cnt)
    write_json(os.path.join(_out_dir(cfg), "format_artifact.json"), out)
    return out


def load_metrics(cfg: dict) -> dict:
    d = _out_dir(cfg)
    out = {}
    for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if fn.endswith(".metrics.json"):
            out[fn[: -len(".metrics.json")]] = read_json(os.path.join(d, fn))
    return out


__all__ = ["ensure_files", "load_items", "sample", "swap_indent", "official_metrics", "run_errorbench",
           "format_artifact_stats", "load_metrics", "parse_critic_output", "json"]
