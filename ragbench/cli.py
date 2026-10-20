"""Linha de comando: ``python -m ragbench <comando> [opções]``.

Caminho mais simples — o painel (roda os testes, mostra as falhas e os dados no navegador):
    python -m ragbench ui

O mesmo pela linha de comando (retoma de onde parou se for interrompido):
    python -m ragbench doctor  --profile quick
    python -m ragbench prepare --profile quick
    python -m ragbench bench   --profile quick          # os dois artigos, em ordem de prioridade
    python -m ragbench report  --profile quick          # relatórios + análise de falhas (FALHAS.md)

Cada artigo também roda sozinho:
    python -m ragbench bench --paper dynakrag  --profile quick
    python -m ragbench bench --paper ragcritic --profile quick
    python -m ragbench dynakrag run  --dataset hotpotqa --method dynakrag --limit 20
    python -m ragbench ragcritic run --dataset nq --method ragcritic --limit 20
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

from .config import PROFILES, load_config


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--profile", default="quick", choices=PROFILES)
    p.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR", help="sobrescreve o config")
    p.add_argument("--root", default=None, help="pasta base (padrão: a pasta atual)")


def _setup(a) -> dict:
    from .core.io import setup_logging

    cfg = load_config(a.profile, a.set, a.root)
    setup_logging(os.path.join(cfg["paths"]["logs"], f"{cfg['profile']}_{time.strftime('%Y%m%d')}.log"))
    if cfg["llm"]["backend"] == "mock":
        from .core.tokens import disable_tokenizer

        disable_tokenizer()
    return cfg


_OPEN_LLMS: list = []          # clientes criados pelo comando em execução; main() fecha todos ao terminar


def _llm(cfg):
    from .core.llm import LLM

    llm = LLM(cfg)
    _OPEN_LLMS.append(llm)
    return llm


def all_datasets(cfg: dict) -> list[str]:
    from .config import PAPERS, paper_datasets

    return list(dict.fromkeys(ds for p in PAPERS for ds in paper_datasets(cfg, p)))


# ============================================================================ comandos
def cmd_doctor(cfg, a):
    from .doctor import run

    sys.exit(1 if run(cfg) else 0)


def cmd_prepare(cfg, a):
    from .config import PAPERS, paper_datasets, retrieval_spec
    from .core.datasets import load_corpus, prepare_dataset
    from .core.io import get_logger
    from .core.retrieval import build_index
    from .core.tokens import count_tokens, tokenizer_kind

    log = get_logger()
    datasets = a.datasets or all_datasets(cfg)
    ok, failed = [], {}
    for ds in datasets:
        try:
            need_train = ds in paper_datasets(cfg, "dynakrag") or ds in cfg["ragcritic"]["errorsystem"]["datasets"]
            prepare_dataset(cfg, ds, force=a.force, need_train=need_train)
            ok.append(ds)
        except Exception as e:  # noqa: BLE001
            failed[ds] = f"{type(e).__name__}: {e}"
            log.error(f"[{ds}] falhou: {e}")
    if not a.no_index:
        papers = [p for p in PAPERS if a.paper in ("all", p)]
        for ds in ok:
            for paper in papers:
                if ds in paper_datasets(cfg, paper):
                    spec = retrieval_spec(cfg, paper)
                    try:
                        build_index(load_corpus(cfg, ds), cfg["paths"]["data"], ds, spec)
                    except Exception as e:  # noqa: BLE001
                        failed[f"{ds}/{paper} índice"] = f"{type(e).__name__}: {e}"
                        log.error(f"[{ds}] índice {spec} falhou: {e}\n  → alternativa: --set retrieval.{paper}.kind=bm25")
    if "ragcritic" in cfg and not a.no_bench_files:
        try:
            from .ragcritic.errorbench import ensure_files

            ensure_files(cfg)
        except Exception as e:  # noqa: BLE001
            failed["rag_error_bench"] = f"{type(e).__name__}: {e}"
    count_tokens("aquecendo o tokenizer")
    log.info(f"contagem de tokens: {tokenizer_kind()}")
    log.info(f"preparados: {ok}" + (f" | FALHAS: {failed}" if failed else ""))
    if failed and not ok:
        sys.exit(1)


def bench_jobs(cfg, llm, paper: str = "all", datasets=None, methods=None) -> list:
    """Jobs dos dois artigos. ``methods`` aceita nomes simples ou 'artigo/método' (como o painel manda)."""
    from .dynakrag import bench as dkb
    from .ragcritic import bench as rcb

    def pick(name: str):
        if not methods:
            return None
        own = [m.split("/", 1)[1] for m in methods if m.startswith(name + "/")]
        own += [m for m in methods if "/" not in m]
        return own or ["__nenhum__"]            # só métodos de outros artigos foram pedidos

    papers = [paper] if isinstance(paper, str) else list(paper or ["all"])
    jobs = []
    for name, mod in (("ragcritic", rcb), ("dynakrag", dkb)):
        if "all" in papers or name in papers:
            jobs += mod.jobs(cfg, llm, datasets, pick(name))
    return jobs


def method_catalog(cfg: dict) -> dict[str, list[str]]:
    """Os nomes que ``bench --methods`` aceita neste perfil, por artigo."""
    from .dynakrag import bench as dkb
    from .ragcritic import bench as rcb

    return {"ragcritic": ["errorbench"] + [n for n, _ in rcb.method_names(cfg)],
            "dynakrag": [n for n, _ in dkb.method_names(cfg)]}


def check_bench_names(cfg: dict, datasets=None, methods=None) -> None:
    """Para na hora se um dataset ou método pedido não existe (antes, a etapa só sumia da fila, sem aviso)."""
    import difflib

    from .core.datasets import DATASET_NAMES

    problems = []
    for ds in datasets or []:
        if ds not in DATASET_NAMES:
            problems.append(f"dataset desconhecido: {ds}   (existem: {', '.join(DATASET_NAMES)})")
    catalog = method_catalog(cfg)
    full = [f"{p}/{n}" for p, names in catalog.items() for n in names]
    simple = {n for names in catalog.values() for n in names}
    bad = [m for m in methods or [] if m not in full and m not in simple]
    for m in bad:
        cands = full if "/" in m else sorted(simple)
        # erro de digitação, ou um nome válido com uma sobra grande depois (texto colado junto)
        guess = next(iter(difflib.get_close_matches(m, cands, n=1, cutoff=0.6)), None) or \
            max((c for c in cands if m.startswith(c)), key=len, default=None)
        problems.append(f"método desconhecido: {m}" + (f"   (você quis dizer {guess}?)" if guess else ""))
    if not problems:
        return
    print("\n".join(problems))
    if bad:
        print(f"métodos do perfil {cfg['profile']}:")
        for p, names in catalog.items():
            print(f"  {p}/  " + ", ".join(names))
    print("nada foi rodado — confira a linha de comando (sobrou algum texto colado no fim?).")
    sys.exit(2)


def cmd_bench(cfg, a):
    from .core.jobs import run_jobs

    check_bench_names(cfg, a.datasets, a.methods)
    llm = _llm(cfg)
    jobs = bench_jobs(cfg, llm, a.paper, a.datasets, a.methods)
    tiers = tuple(a.tier) if a.tier else (1, 2, 3)
    if a.dry_run:
        for j in sorted(jobs, key=lambda j: j.tier):
            if j.tier in tiers:
                print(f"tier {j.tier}  {j.key}  {j.desc}")
        return
    status_path = os.path.join(cfg["paths"]["results"], "status.json")
    if not any(j.tier in tiers for j in jobs):
        from .core.io import get_logger

        get_logger().warning("nenhuma etapa na fila: esta combinação de --paper, --tier, --datasets e --methods não "
                             "escolhe nada (ou os datasets ainda não foram preparados: rode prepare).")
    try:
        run_jobs(jobs, status_path, tiers)
    finally:
        from .core.io import get_logger, write_json

        write_json(os.path.join(cfg["paths"]["results"], "llm_usage.json"), llm.summary())
        if not a.no_report:
            from .report import build_all

            paths = build_all(cfg)
            get_logger().info(f"relatórios: {paths}")


def cmd_report(cfg, a):
    from .report import build_all

    for k, v in build_all(cfg).items():
        print(f"{k}: {v}")


# ----------------------------------------------------------------------------- DynaKRAG
def cmd_dk_train(cfg, a):
    from .dynakrag.bench import ensure_controller, make_env

    llm = _llm(cfg)
    for ds in a.datasets or cfg["dynakrag"]["datasets"]:
        print(ensure_controller(cfg, make_env(cfg, ds, llm), force=a.force))


def cmd_dk_run(cfg, a):
    from .core.stats import summarize
    from .dynakrag.bench import run_method

    llm = _llm(cfg)
    rows = run_method(cfg, a.dataset, a.method, llm, a.limit)
    s = summarize(rows)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})


# ----------------------------------------------------------------------------- RAG-Critic
def cmd_rc_run(cfg, a):
    from .core.stats import summarize
    from .ragcritic.bench import run_method

    rows = run_method(cfg, a.dataset, a.method, _llm(cfg), a.limit)
    s = summarize(rows)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()})


def cmd_rc_errorbench(cfg, a):
    import json

    from .ragcritic.errorbench import format_artifact_stats, run_errorbench

    llm = _llm(cfg)
    print("artefato de formato:", json.dumps(format_artifact_stats(cfg)))
    if a.which == "finetuned":
        res = run_errorbench(cfg, llm, "finetuned")
    else:
        res = run_errorbench(cfg, llm, "zeroshot", a.model or cfg["models"]["generator"])
    for cond, m in res.items():
        if "identification" in m:
            print(cond, "identificação:", {k: round(v, 4) for k, v in m["identification"].items()},
                  "classificação:", {k: round(v, 4) for k, v in m["classification"].items()})
        else:
            print(cond, m)


def cmd_rc_audit(cfg, a):
    import json

    from .ragcritic.errorbench import format_artifact_stats

    print(json.dumps(format_artifact_stats(cfg), indent=1))


def cmd_rc_errorsystem(cfg, a):
    import json

    from .ragcritic.errorsystem.pipeline import run_all

    print(json.dumps(run_all(cfg, _llm(cfg)), indent=1, ensure_ascii=False))


def cmd_rc_build_data(cfg, a):
    from .ragcritic.training.build_data import build

    print(build(cfg, a.source, a.limit, a.k, not a.keep_indent))


def cmd_rc_train(cfg, a):
    rest = list(a.rest or [])
    if rest and rest[0] == "--":
        rest = rest[1:]
    mod = __import__(f"ragbench.ragcritic.training.train_{a.stage}", fromlist=["main"])
    mod.main(rest)


# ----------------------------------------------------------------------------- análise de falhas e painel
def cmd_analyze(cfg, a):
    from .analysis import write_reports

    for k, v in write_reports(cfg).items():
        print(f"{k}: {v}")


def cmd_diagnostico(cfg, a):
    from .diagnostico import format_case, run

    methods = a.metodo or ["ragcritic", "dynakrag"]
    datasets = [a.dataset] if a.dataset else a.datasets
    out = run(cfg, _llm(cfg), methods, datasets, a.alvos, a.id, a.criterio, a.corte, a.limite)
    if a.id:
        if not out["results"]:
            print("nenhuma pergunta com esse id nos resultados da rodada (confira --metodo, --dataset e --profile)")
        for r in out["results"]:
            print(format_case(r) + "\n")
    else:
        n = sum(not r["skipped"] for r in out["results"])
        print(f"{n} pergunta(s) testada(s)")
        for k, v in out["paths"].items():
            print(f"{k}: {v}")
    for note in out["notes"]:
        print(f"aviso: {note}")


def cmd_ui(a):
    """Não passa por ``_setup``: o painel carrega o config de cada perfil conforme você troca no topo da página."""
    from .ui.data import latest_profile
    from .ui.server import export_html, serve

    root = os.path.abspath(a.root or os.getcwd())
    profile = a.profile or latest_profile(root, a.set)       # sem --profile: o que tem resultado mais recente
    if a.export:
        print(export_html(root, profile, a.export, a.set, a.methods))
        return
    serve(root, profile, a.set, port=a.port, open_browser=not a.no_browser)


# ----------------------------------------------------------------------------- smoke
def cmd_smoke(cfg, a):
    """Roda TUDO com o perfil smoke numa pasta temporária e confere as saídas."""
    import tempfile

    from .core import datasets as D
    from .core import retrieval as R
    from .core.io import read_jsonl

    root = a.root or tempfile.mkdtemp(prefix="ragbench_smoke_")
    if os.path.exists(os.path.join(root, "ragbench_results", "smoke")) and not a.keep:
        shutil.rmtree(os.path.join(root, "ragbench_results", "smoke"), ignore_errors=True)
        shutil.rmtree(os.path.join(root, "ragbench_data", "smoke"), ignore_errors=True)
    D.clear_cache()
    R.clear_cache()
    base = ["--profile", "smoke", "--root", root]
    res = os.path.join(root, "ragbench_results", "smoke")
    painel = os.path.join(res, "painel.html")
    for argv in (["prepare"] + base, ["bench"] + base + ["--no-report"], ["ragcritic", "errorsystem"] + base,
                 ["ragcritic", "build-train-data", "--source", "local"] + base, ["report"] + base,
                 ["ui", "--export", painel] + base):
        print(f"\n$ python -m ragbench {' '.join(argv)}")
        main(argv, _exit=False)
    must = [os.path.join(res, "REPORT.md"), os.path.join(res, "dynakrag", "REPORT.md"), os.path.join(res, "ragcritic", "REPORT.md"),
            os.path.join(res, "dynakrag", "toy_multihop", "dynakrag.jsonl"),
            os.path.join(res, "ragcritic", "toy_single", "ragcritic.jsonl"),
            os.path.join(res, "ragcritic", "toy_multihop", "ragcritic.jsonl"), os.path.join(res, "COMPARACAO.md"),
            os.path.join(res, "FALHAS.md"), os.path.join(res, "TABELAS.md"), os.path.join(res, "tabelas.tex"),
            os.path.join(res, "figs", "tabela_dynakrag.png"), os.path.join(res, "figs", "tabela_ragcritic.png"),
            os.path.join(res, "falhas_por_teste.csv"), os.path.join(res, "metrics.csv"),
            os.path.join(res, "comparacao.csv"), painel]
    missing = [m for m in must if not os.path.exists(m)]
    if not missing:
        with open(must[0], encoding="utf-8") as f:
            if "Falha ao gerar" in f.read():
                missing.append("REPORT.md contém 'Falha ao gerar o relatório'")
    import json

    with open(os.path.join(res, "status.json"), encoding="utf-8") as f:
        status = json.load(f)
    failed = {k: v.get("error") for k, v in status.items() if v.get("state") != "done"}
    n_rows = len(read_jsonl(os.path.join(res, "dynakrag", "toy_multihop", "dynakrag.jsonl"))) if not missing else 0
    if missing or failed:
        print(f"\nsmoke FALHOU\n  faltando: {missing}\n  jobs com falha: {failed}")
        sys.exit(1)
    print(f"\nsmoke OK — {len(status)} jobs, {n_rows} perguntas no DynaKRAG toy. Resultados em {res}\n"
          f"Para ver o painel com estes resultados de teste: abra {painel}")


# ============================================================================ parser
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m ragbench",
                                 description="DynaKRAG e RAG-Critic isolados, com Ollama")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="confere Ollama, modelos, GPU e estima o tempo")
    _common(p)
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("prepare", help="baixa datasets, monta corpora, índices e o RAG-Error Bench")
    _common(p)
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--paper", default="all", choices=["all", "dynakrag", "ragcritic"],
                   help="constrói só os índices deste artigo")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-index", action="store_true")
    p.add_argument("--no-bench-files", action="store_true")
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("bench", help="roda os benchmarks em ordem de prioridade (retomável)")
    _common(p)
    p.add_argument("--paper", default=["all"], nargs="+", choices=["all", "dynakrag", "ragcritic"],
                   help="um ou mais artigos (padrão: os dois)")
    p.add_argument("--tier", type=int, nargs="*", choices=[1, 2, 3])
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--methods", nargs="*")
    p.add_argument("--dry-run", action="store_true", help="só lista os jobs")
    p.add_argument("--no-report", action="store_true")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("report", help="gera os relatórios com o que já terminou")
    _common(p)
    p.set_defaults(fn=cmd_report)

    p = sub.add_parser("analyze", help="análise de falhas por pergunta (FALHAS.md + CSV)")
    _common(p)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("diagnostico", help="refaz as falhas de busca e raciocínio com os parágrafos de suporte no contexto")
    _common(p)
    p.add_argument("--metodo", nargs="+", choices=["ragcritic", "dynakrag"], help="padrão: os dois")
    p.add_argument("--dataset", help="um dataset (com --id, para ver uma pergunta)")
    p.add_argument("--datasets", nargs="*", help="restringe os datasets de cada método")
    p.add_argument("--id", nargs="+", help="uma ou mais perguntas, mostradas em detalhe (qualquer etapa)")
    p.add_argument("--alvos", nargs="+", default=["busca", "raciocinio"], choices=["busca", "raciocinio"])
    p.add_argument("--criterio", default="acc", choices=["acc", "em", "f1", "bleu", "bleu1", "rougel"],
                   help="critério de acerto (padrão: acc, o gabarito aparece na resposta)")
    p.add_argument("--corte", type=float, default=0.5, help="corte para f1, bleu, bleu1 e rougel")
    p.add_argument("--limite", type=int, help="testa só as N primeiras perguntas")
    p.set_defaults(fn=cmd_diagnostico)

    p = sub.add_parser("ui", help="abre o painel no navegador (rodar testes, ver falhas, explorar os dados)")
    p.add_argument("--profile", default=None, choices=PROFILES, help="perfil aberto de início (padrão: o que tem "
                   "resultado mais recente); dá para trocar no topo da página")
    p.add_argument("--set", action="append", default=[], metavar="CHAVE=VALOR", help="sobrescreve o config")
    p.add_argument("--root", default=None, help="pasta base (padrão: a pasta atual)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="não abre o navegador sozinho")
    p.add_argument("--export", metavar="ARQUIVO.html", help="gera um HTML único com os resultados (sem servidor)")
    p.add_argument("--methods", nargs="*", help="com --export: métodos cujo passo a passo entra no arquivo "
                   "(ex.: ragcritic/ragcritic dynakrag/dynakrag)")
    p.set_defaults(fn=cmd_ui)

    p = sub.add_parser("smoke", help="teste ponta a ponta com LLM falso (sem GPU/Ollama/internet)")
    p.add_argument("--root", default=None)
    p.add_argument("--keep", action="store_true")
    p.set_defaults(fn=cmd_smoke, profile="smoke", set=[])

    # DynaKRAG
    dk = sub.add_parser("dynakrag", help="comandos do DynaKRAG").add_subparsers(dest="sub", required=True)
    p = dk.add_parser("train", help="rótulos da Eq. 1 + RF de valor + modelo de continuação")
    _common(p)
    p.add_argument("--datasets", nargs="*")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_dk_train)
    p = dk.add_parser("run", help="roda um método num dataset")
    _common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_dk_run)

    # RAG-Critic
    rc = sub.add_parser("ragcritic", help="comandos do RAG-Critic").add_subparsers(dest="sub", required=True)
    p = rc.add_parser("run", help="roda um método/variante num dataset")
    _common(p)
    p.add_argument("--dataset", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_rc_run)
    p = rc.add_parser("errorbench", help="RAG-Error Bench (+ auditoria de formato)")
    _common(p)
    p.add_argument("--which", default="finetuned", choices=["finetuned", "zeroshot"])
    p.add_argument("--model", default=None)
    p.set_defaults(fn=cmd_rc_errorbench)
    p = rc.add_parser("audit", help="mostra o artefato de formato do benchmark oficial")
    _common(p)
    p.set_defaults(fn=cmd_rc_audit)
    p = rc.add_parser("errorsystem", help="etapas 1–3: constrói um sistema de erros local")
    _common(p)
    p.set_defaults(fn=cmd_rc_errorsystem)
    p = rc.add_parser("build-train-data", help="dados de SFT e DPO coarse-to-fine para o crítico")
    _common(p)
    p.add_argument("--source", default="official", choices=["official", "local"])
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--k", type=int, default=1, help="negativos finos por exemplo")
    p.add_argument("--keep-indent", action="store_true", help="não neutraliza o recuo dos prompts")
    p.set_defaults(fn=cmd_rc_build_data)
    p = rc.add_parser("train", help="treina o crítico (requer requirements-train.txt)")
    _common(p)
    p.add_argument("stage", choices=["sft", "dpo"])
    p.add_argument("rest", nargs=argparse.REMAINDER, help="argumentos do script de treino")
    p.set_defaults(fn=cmd_rc_train)
    return ap


def main(argv=None, _exit: bool = True):
    for stream in (sys.stdout, sys.stderr):  # console do Windows sem UTF-8: não quebrar por causa de "✔", "τ" etc.
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    a = build_parser().parse_args(argv)
    if a.cmd == "smoke":
        return a.fn(None, a)
    if a.cmd == "ui":
        return a.fn(a)
    from .core.llm import LLMError

    cfg = _setup(a)
    try:
        return a.fn(cfg, a)
    except SystemExit as e:
        if _exit:
            raise
        if e.code:
            raise
    except LLMError as e:       # servidor fechado, modelo ausente, queda longa: o motivo em uma linha, sem traceback
        if not _exit:
            raise
        print(f"\nparou: {e}\nO que já terminou está salvo: rode o mesmo comando de novo para continuar de onde parou.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrompido — rode o mesmo comando de novo para continuar de onde parou.")
        if _exit:
            sys.exit(130)
    finally:                    # fecha o cache e a conexão HTTP (sem isso o Python avisa de socket aberto ao sair)
        while _OPEN_LLMS:
            try:
                _OPEN_LLMS.pop().close()
            except Exception:
                pass
