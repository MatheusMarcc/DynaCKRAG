"""Linha de comando: python -m dynacritic <comando> [opções]

Ordem típica:
  doctor → prepare → index → baseline → run --policy uniform → label → train → run --policy rf
  → critic → repair --mode menu|code|state → eval
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time

from .config import load_config


def _env(cfg):
    from .env import Env
    from .llm import LLM
    from .retrieval import load_retriever

    return Env(cfg, load_retriever(cfg), LLM(cfg))


# ------------------------------------------------------------------ doctor
def cmd_doctor(cfg, a):
    import httpx

    base = cfg["llm"]["base_url"].rstrip("/")
    print(f"backend: {cfg['llm']['backend']}  |  base_url: {base}  |  num_ctx: {cfg['llm']['num_ctx']}")
    if cfg["llm"]["backend"] != "ollama":
        print("doctor só verifica o backend ollama.")
        return
    try:
        tags = httpx.get(f"{base}/api/tags", timeout=10).json()
    except Exception as e:
        print(f"[ERRO] Ollama não respondeu em {base}: {e}\n  → abra o app do Ollama ou rode `ollama serve`.")
        return
    have = {m["name"].lower() for m in tags.get("models", [])}
    for role, model in cfg["models"].items():
        ok = model.lower() in have or f"{model}:latest".lower() in have
        print(f"[{'ok' if ok else 'FALTA'}] {role:9s} {model}" + ("" if ok else f"   → ollama pull {model}"))
        if not ok:
            continue
        t0 = time.time()
        r = httpx.post(f"{base}/api/chat", timeout=600, json={
            "model": model, "stream": False, "keep_alive": "5m",
            "messages": [{"role": "user", "content": "Say 'ready' and nothing else."}],
            "options": {"temperature": 0, "num_ctx": cfg["llm"]["num_ctx"], "num_predict": 16}}).json()
        dt = time.time() - t0
        tps = r.get("eval_count", 0) / max(1e-9, r.get("eval_duration", 1) / 1e9)
        ps = httpx.get(f"{base}/api/ps", timeout=10).json().get("models", [])
        info = next((m for m in ps if model.lower() in (str(m.get("name")).lower(), str(m.get("model")).lower())), None)
        gpu = ""
        if info and info.get("size"):
            frac = info.get("size_vram", 0) / info["size"]
            gpu = f" | {frac:.0%} na GPU ({info.get('size_vram', 0) / 2**30:.1f} de {info['size'] / 2**30:.1f} GiB)"
        print(f"        resposta={r.get('message', {}).get('content', '').strip()[:30]!r} "
              f"| {tps:.1f} tok/s | 1ª chamada {dt:.1f}s{gpu}")


# ------------------------------------------------------------------ comandos principais
def cmd_prepare(cfg, a):
    from .data import prepare

    prepare(cfg, a.source, a.train_json, a.dev_json)


def cmd_index(cfg, a):
    from .retrieval import build_index

    build_index(cfg)


def cmd_baseline(cfg, a):
    from .loop import run_fixed_k

    env = _env(cfg)
    print(run_fixed_k(cfg, env, a.split, a.k, a.name or f"fixedk{a.k}", a.limit))
    print(json.dumps(env.llm.summary(), indent=1))


def cmd_run(cfg, a):
    from .controller import make_policy
    from .loop import run_loop_a

    env = _env(cfg)
    pol = make_policy(cfg, a.policy)
    comp = cfg["loop"]["compression"] and not a.no_compress
    name = a.name or f"dyna_{pol.name}" + ("" if comp else "_nocomp")
    print(run_loop_a(cfg, env, pol, a.split, name, comp, a.limit))
    print(json.dumps(env.llm.summary(), indent=1))


def cmd_label(cfg, a):
    from .labeling import label_offline

    env = _env(cfg)
    label_offline(cfg, env, a.limit)
    print(json.dumps(env.llm.summary(), indent=1))


def cmd_train(cfg, a):
    from .labeling import train_controller

    train_controller(cfg)


def cmd_critic(cfg, a):
    from .critic import run_critic
    from .llm import LLM

    run_critic(cfg, LLM(cfg), a.run)


def cmd_critic_bench(cfg, a):
    from .critic import critic_bench
    from .llm import LLM

    critic_bench(LLM(cfg), a.file, a.n)


def cmd_repair(cfg, a):
    from .controller import make_policy
    from .repair import run_repair

    env = _env(cfg)
    mode = a.mode or cfg["repair"]["mode"]
    pol_name = a.policy or ("rf" if os.path.exists(cfg["paths"]["rf"]) else "uniform")
    pol = make_policy(cfg, pol_name) if mode == "state" else None
    print(run_repair(cfg, env, a.run, mode, pol))
    print(json.dumps(env.llm.summary(), indent=1))


def cmd_eval(cfg, a):
    from .evaluate import evaluate_runs

    evaluate_runs(cfg)


def cmd_smoke(cfg, a):
    """Roda o pipeline inteiro com dados sintéticos + LLM mock + BM25. Não precisa de GPU."""
    from .toy import make_toy

    over = ["data.dir=data_smoke", "llm.backend=mock", "retriever.kind=bm25", "llm.cache_path=:memory:",
            "data.n_train=80", "data.n_dev=40"]
    if os.path.isdir("data_smoke"):
        shutil.rmtree("data_smoke")
    cfg = load_config(a.config, over)
    tr, dv = make_toy("data_smoke/raw")
    steps = [
        (cmd_prepare, argparse.Namespace(source="json", train_json=tr, dev_json=dv)),
        (cmd_index, None),
        (cmd_baseline, argparse.Namespace(split="dev", k=3, name="fixedk3", limit=None)),
        (cmd_run, argparse.Namespace(policy="uniform", split="dev", name=None, no_compress=False, limit=None)),
        (cmd_label, argparse.Namespace(limit=None)),
        (cmd_train, None),
        (cmd_run, argparse.Namespace(policy="rf", split="dev", name=None, no_compress=False, limit=None)),
        (cmd_critic, argparse.Namespace(run="dyna_rf")),
        (cmd_repair, argparse.Namespace(run="dyna_rf", mode="menu", policy=None)),
        (cmd_repair, argparse.Namespace(run="dyna_rf", mode="code", policy=None)),
        (cmd_repair, argparse.Namespace(run="dyna_rf", mode="state", policy="rf")),
        (cmd_eval, None),
    ]
    for fn, ns in steps:
        print(f"\n===== {fn.__name__.replace('cmd_', '')} =====")
        fn(cfg, ns)
    print("\nsmoke OK — o encanamento funciona. Agora rode `doctor` e troque para o Ollama.")


# ------------------------------------------------------------------ parser
def main(argv=None):
    import sys

    for stream in (sys.stdout, sys.stderr):  # acentos/Δ/→ sem erro no console do Windows
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS)
    common.add_argument("--set", action="append", default=argparse.SUPPRESS, metavar="CHAVE=VALOR",
                        help="sobrescreve o config, ex.: --set models.generator=qwen2.5:7b")
    p = argparse.ArgumentParser(prog="dynacritic", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR",
                   help="sobrescreve o config, ex.: --set models.generator=qwen2.5:7b")
    sub = p.add_subparsers(dest="cmd", required=True)
    _orig_add = sub.add_parser
    sub.add_parser = lambda *args, **kw: _orig_add(*args, parents=[common], **kw)  # type: ignore

    sub.add_parser("doctor", help="verifica Ollama, modelos, tok/s e quanto do modelo ficou na GPU")
    sp = sub.add_parser("prepare", help="baixa/amostra HotpotQA e monta o corpus reduzido")
    sp.add_argument("--source", choices=["hf", "json"], default="hf")
    sp.add_argument("--train-json"); sp.add_argument("--dev-json")
    sub.add_parser("index", help="constrói o índice (BM25 ou denso)")
    sp = sub.add_parser("baseline", help="RAG de K fixo")
    sp.add_argument("--k", type=int, default=5); sp.add_argument("--split", default="dev")
    sp.add_argument("--name"); sp.add_argument("--limit", type=int)
    sp = sub.add_parser("run", help="laço A (DynaKRAG)")
    sp.add_argument("--policy", choices=["uniform", "rf"]); sp.add_argument("--split", default="dev")
    sp.add_argument("--name"); sp.add_argument("--no-compress", action="store_true"); sp.add_argument("--limit", type=int)
    sp = sub.add_parser("label", help="rotulagem offline ΔSR nas perguntas de treino")
    sp.add_argument("--limit", type=int)
    sub.add_parser("train", help="treina a random forest (modelo de valor de ação)")
    sp = sub.add_parser("critic", help="roda o RAG-Critic-3B sobre uma execução")
    sp.add_argument("--run", required=True)
    sp = sub.add_parser("critic-bench", help="valida o crítico no RAG-Error Bench oficial")
    sp.add_argument("--file", required=True); sp.add_argument("--n", type=int)
    sp = sub.add_parser("repair", help="laço B (RAG-Critic): reparo das respostas marcadas como erro")
    sp.add_argument("--run", required=True); sp.add_argument("--mode", choices=["menu", "code", "state"])
    sp.add_argument("--policy", choices=["uniform", "rf"])
    sub.add_parser("eval", help="tabela comparativa de todas as execuções")
    sub.add_parser("smoke", help="teste ponta a ponta com dados sintéticos e LLM mock")

    a = p.parse_args(argv)
    cfg = load_config(a.config, a.set)
    fn = {"doctor": cmd_doctor, "prepare": cmd_prepare, "index": cmd_index, "baseline": cmd_baseline,
          "run": cmd_run, "label": cmd_label, "train": cmd_train, "critic": cmd_critic,
          "critic-bench": cmd_critic_bench, "repair": cmd_repair, "eval": cmd_eval, "smoke": cmd_smoke}[a.cmd]
    fn(cfg, a)


if __name__ == "__main__":
    main()
