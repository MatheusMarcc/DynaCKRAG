"""Baseline "LLM base": o mesmo gerador respondendo às mesmas perguntas, sem nenhum método e sem recuperação.

Não consta nos dois artigos; foi pedido pelo orientador como referência comum para comparar o RAG-Critic e o
DynaKRAG ("pra base comparando dynakrag e [...] seria o llm base"). Por isso fica fora das tabelas dos artigos:
o comando grava a rodada em ``<results>/llm_base/<dataset>.jsonl`` e um relatório próprio, ``LLM_BASE.md``,
que põe o LLM base ao lado dos métodos já rodados, nas mesmas perguntas.

* Gerador: ``models.generator`` da configuração (o mesmo dos dois métodos).
* Prompt: o de resposta sem recuperação do pacote (``NO_RETRIEVAL_*`` do DynaKRAG, o mesmo que o Adaptive-RAG
  usa na rota "sem retrieval"): só a pergunta, pedindo a resposta curta.
* Perguntas: as de avaliação de cada dataset, na quantidade do artigo que usa o dataset (o maior ``n_eval``,
  quando os dois usam).
* Métricas: EM e F1 (as dos artigos), BLEU e Levenshtein normalizado (as adicionais do diagnóstico).
"""
from __future__ import annotations

import csv
import os

from .config import PAPERS, paper_datasets
from .core.datasets import DATASET_NAMES, is_prepared, load_records
from .core.io import ensure_dir, get_logger, read_jsonl
from .core.report import md_table
from .core.runner import run_records
from .core.text import all_scores, extract_final_answer
from .dynakrag import prompts as DKP

METRICS = ("em", "f1", "bleu", "lev")
METRIC_NAMES = {"em": "EM", "f1": "F1", "bleu": "BLEU", "lev": "Lev"}
# métodos já rodados que entram na comparação: (artigo, arquivo, nome); Fixed-K = o melhor K da grade, por F1
COMPARE = [("ragcritic", "standard_rag", "Standard RAG (top-5, Prompt A)"),
           ("ragcritic", "ragcritic", "RAG-Critic"),
           ("dynakrag", "fixed_k*", "Fixed-K (melhor K)"),
           ("dynakrag", "dynakrag", "DynaKRAG")]
RETRIEVER = {"ragcritic": "E5-base-v2", "dynakrag": "BGE-large"}


def datasets_and_n(cfg: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in PAPERS:
        for ds in paper_datasets(cfg, p):
            out[ds] = max(out.get(ds, 0), int(cfg[p]["n_eval"]))
    return out


def answer(llm, question: str, max_tokens: int) -> tuple[str, str]:
    raw = llm.chat("generator", [{"role": "system", "content": DKP.NO_RETRIEVAL_SYSTEM},
                                 {"role": "user", "content": DKP.NO_RETRIEVAL_USER.format(question=question)}],
                   task="no_retrieval_answer", max_tokens=max_tokens)
    return extract_final_answer(raw), raw


def result_path(cfg: dict, dataset: str) -> str:
    return os.path.join(cfg["paths"]["results"], "llm_base", f"{dataset}.jsonl")


def run_dataset(cfg: dict, llm, dataset: str, n: int, limit: int | None = None) -> list[dict]:
    recs = load_records(cfg, dataset, "eval", n)
    if limit:
        recs = recs[:limit]
    max_tokens = int(cfg["dynakrag"]["answer_max_tokens"])

    def fn(rec):
        pred, raw = answer(llm, rec["question"], max_tokens)
        return {"pred": pred, "raw": raw, "n_ret": 0, "doc_titles": []}

    return run_records(recs, fn, result_path(cfg, dataset), llm, desc=f"[LLM base·{dataset}]",
                       workers=int(cfg["llm"].get("workers", 1)))


# ============================================================================ comparação
def _scores(row: dict) -> dict:
    return all_scores(str(row.get("pred") or ""), row.get("answers") or [])


def _mean(xs) -> float | None:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _load(cfg: dict, paper: str, dataset: str, name: str) -> tuple[str | None, dict]:
    base = os.path.join(cfg["paths"]["results"], paper, dataset)
    if name.endswith("*"):
        best, best_f1, best_rows = None, -1.0, {}
        for k in cfg["dynakrag"]["baselines"]["fixed_k_grid"]:
            rows = {r["id"]: r for r in read_jsonl(os.path.join(base, f"fixed_k{k}.jsonl")) if not r.get("error")}
            f1 = _mean(r["f1"] for r in rows.values())
            if rows and f1 > best_f1:
                best, best_f1, best_rows = f"fixed_k{k}", f1, rows
        return best, best_rows
    rows = {r["id"]: r for r in read_jsonl(os.path.join(base, f"{name}.jsonl")) if not r.get("error")}
    return (name if rows else None), rows


def _fmt(x: float | None, pct: bool = True) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}%" if pct else f"{x:.0f}"


def write_report(cfg: dict, done: dict[str, list[dict]]) -> dict:
    out_dir = ensure_dir(os.path.join(cfg["paths"]["results"], "llm_base"))
    gen = cfg["models"]["generator"]
    md = ["# LLM base: o gerador sem método e sem recuperação", "",
          f"Gerador `{gen}` (o mesmo dos dois métodos) respondendo só com a pergunta, com o prompt de resposta sem "
          "recuperação do pacote. Este baseline não consta nos artigos; foi pedido pelo orientador como referência "
          "comum. As linhas dos métodos são as rodadas já gravadas, restritas às mesmas perguntas do LLM base.", "",
          "Métricas sobre o texto normalizado, com o máximo entre as respostas de referência: EM e F1 (as dos "
          "artigos), BLEU (`sentence_bleu` do sacreBLEU) e Lev = similaridade de Levenshtein normalizada "
          "(1 − distância / maior comprimento, em caracteres). Valores em %, média por pergunta.", ""]
    csv_rows = []
    for ds, base_rows in done.items():
        by_id = {r["id"]: r for r in base_rows}
        rows = [["LLM base (sem recuperação)", "—", str(len(by_id))]
                + [_fmt(_mean(_scores(r)[k] for r in by_id.values())) for k in METRICS]
                + [_fmt(_mean(r.get("tokens", 0) for r in by_id.values()), pct=False)]]
        csv_rows.append([ds, "llm_base", "—", len(by_id)] + [round(_mean(_scores(r)[k] for r in by_id.values()), 4) for k in METRICS])
        for paper, name, pretty in COMPARE:
            if ds not in paper_datasets(cfg, paper):
                continue
            used, mrows = _load(cfg, paper, ds, name)
            ids = [i for i in by_id if i in mrows]
            if not used or not ids:
                continue
            label = pretty + (f" — K = {used[len('fixed_k'):]}" if name.endswith("*") else "")
            sc = [_scores(mrows[i]) for i in ids]
            rows.append([label, RETRIEVER[paper], str(len(ids))] + [_fmt(_mean(s[k] for s in sc)) for k in METRICS]
                        + [_fmt(_mean(mrows[i].get("tokens", 0) for i in ids), pct=False)])
            csv_rows.append([ds, used, RETRIEVER[paper], len(ids)] + [round(_mean(s[k] for s in sc), 4) for k in METRICS])
        md += [f"## {DATASET_NAMES.get(ds, ds)}", "",
               md_table(["Método", "Retriever", "Perguntas"] + [METRIC_NAMES[k] for k in METRICS] + ["Tokens/perg."],
                        rows, "lll" + "r" * (len(METRICS) + 1)), ""]
    md += ["Os retrievers diferem entre os artigos (RAG-Critic: E5-base-v2; DynaKRAG: BGE-large), como em cada "
           "artigo. O LLM base não usa retriever, então é a mesma rodada nas duas comparações.", ""]
    paths = {"relatorio": os.path.join(out_dir, "LLM_BASE.md"), "csv": os.path.join(out_dir, "llm_base.csv")}
    with open(paths["relatorio"], "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(paths["csv"], "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["dataset", "metodo", "retriever", "perguntas"] + list(METRICS))
        w.writerows(csv_rows)
    return paths


def run(cfg: dict, llm, datasets: list[str] | None = None, limit: int | None = None) -> dict:
    log = get_logger()
    plan = datasets_and_n(cfg)
    if datasets:
        plan = {d: plan.get(d, int(cfg["dynakrag"]["n_eval"])) for d in datasets}
    done, notes = {}, []
    for ds, n in plan.items():
        if not is_prepared(cfg, ds):
            notes.append(f"{ds} não está preparado — pulando (rode prepare)")
            continue
        done[ds] = run_dataset(cfg, llm, ds, n, limit)
        log.info(f"LLM base · {ds}: {len(done[ds])} pergunta(s)")
    return {"done": done, "notes": notes, "paths": write_report(cfg, done) if done else {}}
