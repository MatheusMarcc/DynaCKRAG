"""Diagnóstico das falhas de busca e de raciocínio: cada pergunta é respondida de novo com os parágrafos de
suporte do gabarito no contexto.

Não faz parte de nenhum dos dois artigos. É um teste de apoio à análise de falhas (``analysis.py``), que separa
as perguntas em busca / raciocínio / formato. Aqui são reexaminadas só as de **busca** e **raciocínio**.

Condições (o método recebe os documentos já prontos, sem buscar):

* ``suporte``       — só os parágrafos de suporte do gabarito;
* ``suporte+busca`` — os documentos que o método recuperou na rodada, com os parágrafos de suporte que faltavam
  no lugar dos últimos documentos que não são de suporte (o número de documentos não muda); a posição de cada
  parágrafo inserido é sorteada com semente fixa por pergunta.

A geração é a do próprio método:

* RAG-Critic — o gerador π_α com o Prompt A (o mesmo do Standard RAG). Na rodada, o crítico aprovou todas as
  respostas, então a resposta final do RAG-Critic é a do gerador;
* DynaKRAG — a etapa final do método: compressão da evidência e resposta (``env.compress`` e ``env.answer``).

Rótulo de cada pergunta, pelo critério de acerto escolhido (padrão: BLEU ≥ 0,5; o corte foi calibrado contra
julgamento manual de 161 pares resposta-gabarito, ver ``docs/USO.md``):

=========================  ==============  ======================================================
``suporte``                ``suporte+busca``  rótulo
=========================  ==============  ======================================================
acerta                     acerta          resolve com os documentos certos
acerta                     erra            os demais documentos atrapalham
erra                       acerta          inconsistente
erra                       erra            erra mesmo com os documentos certos
=========================  ==============  ======================================================
"""
from __future__ import annotations

import csv
import os
import random
import re
import zlib

from .analysis import (DATASET_NAMES, corpus_index, dataset_order, load_eval, make_cell, read_jsonl, result_files,
                       row_docs, stage_of)
from .config import paper_datasets
from .core.io import ensure_dir, get_logger, write_json, write_jsonl
from .core.report import md_table
from .core.text import all_scores, doc_contents, extract_final_answer, norm_title, normalize_answer

METHODS = {"ragcritic": "ragcritic/ragcritic", "dynakrag": "dynakrag/dynakrag"}
METHOD_NAMES = {"ragcritic": "RAG-Critic", "dynakrag": "DynaKRAG"}
FALLBACK = {"ragcritic": "ragcritic/standard_rag"}      # documentos do RAG-Critic = os do Standard RAG
TARGETS = ("busca", "raciocinio")
CONDITIONS = ("suporte", "suporte+busca")
METRICS = ("em", "f1", "bleu", "lev")           # EM e F1 são as métricas dos artigos; BLEU e Levenshtein são as adicionais
METRIC_NAMES = {"em": "EM", "f1": "F1", "bleu": "BLEU", "lev": "Lev"}
CORTE_PADRAO = 0.5
# corte padrão de cada critério, calibrado contra o julgamento manual de 161 pares resposta-gabarito do perfil mini
# (7B): BLEU ≥ 0,5 concorda em 145 (90,1%); Levenshtein normalizado ≥ 0,7 em 147 (91,3%); EM = 1 em 141 (87,6%)
CORTES = {"bleu": 0.5, "f1": 0.5, "lev": 0.7}
STAGE_NAMES = {"busca": "busca", "raciocinio": "raciocínio"}
LABELS = {
    (True, True): "resolve com os documentos certos",
    (True, False): "os demais documentos atrapalham",
    (False, True): "inconsistente",
    (False, False): "erra mesmo com os documentos certos",
}
_WORD = re.compile(r"\w+")


# ============================================================================ critério de acerto
def is_ok(scores: dict, criterio: str = "bleu", corte: float | None = None) -> bool:
    """Acerto pelo critério escolhido: ``em`` vale 0 ou 1; ``f1``, ``bleu`` e ``lev`` usam o corte (padrão: CORTES)."""
    corte = CORTES.get(criterio, CORTE_PADRAO) if corte is None else corte
    v = float(scores.get(criterio, 0.0))
    return v >= 1.0 if criterio == "em" else v >= corte


def criterio_text(criterio: str, corte: float | None) -> str:
    if criterio == "em":
        return "EM = 1"
    corte = CORTES.get(criterio, CORTE_PADRAO) if corte is None else corte
    return f"{METRIC_NAMES[criterio]} ≥ {str(corte).replace('.', ',')}"


# ============================================================================ documentos do gabarito
def _overlap(text: str, question: str) -> int:
    q = {w for w in _WORD.findall(question.lower()) if len(w) > 2}
    return len(q & set(_WORD.findall(text.lower())))


def gold_docs(rec: dict, corpus: dict, max_docs: int = 6) -> list[dict]:
    """Parágrafos de suporte do gabarito, achados no corpus pelo título.

    Um título pode ter mais de um parágrafo no corpus (MuSiQue repete títulos; o TriviaQA corta a página em
    trechos). De cada título fica um parágrafo: o que contém o gabarito, se houver, senão o de maior sobreposição
    de palavras com a pergunta. Se nenhum parágrafo escolhido contém o gabarito, entra também o melhor que contém.
    """
    golds = [normalize_answer(a) for a in rec.get("answers") or [] if normalize_answer(a)]
    question = rec.get("question", "")

    def has_answer(d: dict) -> bool:
        t = normalize_answer(f"{d.get('title', '')} {d.get('text', '')}")
        return any(g in t for g in golds if g not in ("yes", "no"))

    def rank(d: dict) -> tuple:
        return (has_answer(d), _overlap(d.get("text", ""), question))

    chosen, pool = [], []
    for t in rec.get("supporting_titles") or []:
        cands = corpus["by_title"].get(norm_title(t), [])
        if not cands:
            continue
        pool.extend(cands)
        chosen.append(max(cands, key=rank))
    if golds and pool and not any(has_answer(d) for d in chosen):
        best = [d for d in pool if has_answer(d)]
        if best:
            chosen.append(max(best, key=rank))
    out, seen = [], set()
    for d in chosen:
        if d["id"] not in seen:
            seen.add(d["id"])
            out.append(d)
    return out[:max_docs]


def mixed_docs(retrieved: list[dict], gold: list[dict], qid: str) -> list[dict]:
    """Documentos da rodada com os parágrafos de suporte que faltavam; o total de documentos não muda."""
    gold_ids = {d["id"] for d in gold}
    gold_titles = {norm_title(d["title"]) for d in gold}
    present = {d["id"] for d in retrieved} | {norm_title(d["title"]) for d in retrieved}
    missing = [d for d in gold if d["id"] not in present]
    docs = list(retrieved)
    for _ in missing:                                   # abre espaço tirando o último documento que não é de suporte
        for j in range(len(docs) - 1, -1, -1):
            if docs[j]["id"] not in gold_ids and norm_title(docs[j]["title"]) not in gold_titles:
                del docs[j]
                break
    rng = random.Random(zlib.crc32(str(qid).encode("utf-8")))
    for d in missing:
        docs.insert(rng.randint(0, len(docs)), d)
    return docs


# ============================================================================ geração pelo método
def answer_ragcritic(cfg: dict, llm, question: str, docs: list[dict]) -> dict:
    from .ragcritic import prompts as P

    prompt = P.PROMPT_A.format(passages=P.passages_block([doc_contents(d) for d in docs]), question=question)
    raw = llm.chat("generator", [{"role": "user", "content": prompt}], task="rc_generate",
                   max_tokens=int(cfg["ragcritic"]["generator_max_tokens"]))
    return {"pred": extract_final_answer(raw), "raw": raw}


def answer_dynakrag(cfg: dict, llm, dataset: str, question: str, docs: list[dict]) -> dict:
    from .dynakrag.env import DynaEnv
    from .dynakrag.state import EvidenceState

    env = DynaEnv(cfg, dataset, None, llm)              # sem retriever: os documentos já vêm prontos
    docs = docs[: env.max_final]
    evidence, comp_raw = env.compress(EvidenceState(qid="", question=question), docs)
    pred, raw = env.answer(question, evidence)
    return {"pred": pred, "raw": raw, "evidence": evidence, "compression_raw": comp_raw}


def answer(cfg: dict, llm, method: str, dataset: str, question: str, docs: list[dict]) -> dict:
    if method == "ragcritic":
        return answer_ragcritic(cfg, llm, question, docs)
    return answer_dynakrag(cfg, llm, dataset, question, docs)


# ============================================================================ seleção das perguntas
def select_cases(cfg: dict, methods: list[str], datasets: list[str] | None, targets: list[str],
                 ids: list[str] | None = None) -> tuple[list[dict], list[str]]:
    """Falhas de busca/raciocínio de cada método, com a resposta da rodada e os documentos que ele usou."""
    cmp = cfg.get("compare") or {}
    metric, thr = cmp.get("fail_metric", "em"), float(cmp.get("fail_threshold", 0.5))
    files = result_files(cfg)
    cases, notes = [], []
    for m in methods:
        key = METHODS[m]
        dss = [d for d in dataset_order(paper_datasets(cfg, m)) if (not datasets or d in datasets)]
        for ds in dss:
            path = files.get(ds, {}).get(key)
            if not path:
                notes.append(f"{METHOD_NAMES[m]} · {DATASET_NAMES.get(ds, ds)}: sem resultados da rodada")
                continue
            recs, corpus = load_eval(cfg, ds), corpus_index(cfg, ds)
            if not recs or corpus is None:
                notes.append(f"{DATASET_NAMES.get(ds, ds)}: dados ausentes nesta máquina (rode prepare)")
                continue
            rows = {r["id"]: r for r in read_jsonl(path) if not r.get("error")}
            fb_path = files.get(ds, {}).get(FALLBACK.get(m, ""), "")
            fb = {r["id"]: r for r in read_jsonl(fb_path)} if fb_path else {}
            for qid, row in rows.items():
                if ids and qid not in ids:
                    continue
                rec = recs.get(qid)
                if rec is None:
                    continue
                cell = make_cell(row, rec, corpus, fb.get(qid))
                stage = stage_of(cell, metric, thr)
                if stage not in targets and not ids:
                    continue
                base = fb.get(qid)
                if base is not None and not row.get("changed"):
                    retrieved = row_docs(base, corpus)      # resposta não alterada: os documentos do Standard RAG
                else:
                    retrieved = row_docs(row, corpus, base)
                if m == "dynakrag":
                    retrieved = retrieved[: int(cfg["dynakrag"]["max_final_docs"])]
                cases.append({"method": m, "dataset": ds, "id": qid, "rec": rec, "row": row, "stage": stage,
                              "retrieved": retrieved, "corpus": corpus})
    return cases, notes


# ============================================================================ uma pergunta
def run_case(cfg: dict, llm, case: dict, criterio: str = "bleu", corte: float | None = None) -> dict:
    rec, m, ds = case["rec"], case["method"], case["dataset"]
    golds = rec.get("answers") or []
    gold = gold_docs(rec, case["corpus"])
    out = {"method": m, "dataset": ds, "id": case["id"], "question": rec["question"], "gold": golds,
           "supporting_titles": rec.get("supporting_titles") or [], "stage": case["stage"],
           "original": {"pred": str(case["row"].get("pred") or ""), "docs": [d["title"] for d in case["retrieved"]],
                        "scores": all_scores(str(case["row"].get("pred") or ""), golds)},
           "conditions": {}, "label": None, "skipped": None}
    if not gold:
        out["skipped"] = "os parágrafos de suporte não foram encontrados no corpus"
        return out
    gold_ids = {d["id"] for d in gold}
    docs_by = {"suporte": gold, "suporte+busca": mixed_docs(case["retrieved"], gold, case["id"])}
    for cond in CONDITIONS:
        docs = docs_by[cond]
        res = answer(cfg, llm, m, ds, rec["question"], docs)
        sc = all_scores(res["pred"], golds)
        out["conditions"][cond] = {**res, "docs": [d["title"] for d in docs],
                                   "is_support": [d["id"] in gold_ids for d in docs], "scores": sc,
                                   "ok": is_ok(sc, criterio, corte)}
    out["label"] = LABELS[(out["conditions"]["suporte"]["ok"], out["conditions"]["suporte+busca"]["ok"])]
    return out


def format_case(r: dict) -> str:
    """Saída detalhada de uma pergunta, para o terminal."""
    def sc(s: dict) -> str:
        return " · ".join(f"{METRIC_NAMES[k]} {s[k]:.2f}" for k in METRICS)

    lines = [f"{METHOD_NAMES[r['method']]} · {DATASET_NAMES.get(r['dataset'], r['dataset'])} · {r['id']}",
             f"Pergunta: {r['question']}", f"Gabarito: {' | '.join(r['gold'])}",
             f"Parágrafos de suporte: {' ; '.join(r['supporting_titles']) or '—'}",
             f"Etapa da falha na rodada: {STAGE_NAMES.get(r['stage'], r['stage'])}",
             "", "[rodada original]", f"  documentos: {' ; '.join(r['original']['docs'])}",
             f"  resposta:   {r['original']['pred']}", f"  métricas:   {sc(r['original']['scores'])}"]
    if r["skipped"]:
        return "\n".join(lines + ["", f"Não testada: {r['skipped']}."])
    for cond, c in r["conditions"].items():
        docs = " ; ".join(("★ " if s else "") + t for t, s in zip(c["docs"], c["is_support"]))
        lines += ["", f"[{cond}]", f"  documentos (★ = suporte): {docs}"]
        if c.get("evidence") is not None:
            ev = c["evidence"].replace("\n", "\n              ")
            lines.append(f"  evidência comprimida: {ev}")
        raw = (c["raw"] or "").strip().replace("\n", "\n              ")
        lines += [f"  saída do modelo: {raw}", f"  resposta:   {c['pred']}", f"  métricas:   {sc(c['scores'])}",
                  f"  acertou:    {'sim' if c['ok'] else 'não'}"]
    lines += ["", f"Rótulo: {r['label']}"]
    return "\n".join(lines)


# ============================================================================ relatório
def _mean(xs) -> str:
    xs = list(xs)
    return f"{sum(xs) / len(xs):.3f}" if xs else "—"


def _clip(s: str, n: int = 70) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1] + "…"


def write_report(cfg: dict, results: list[dict], notes: list[str], criterio: str, corte: float, out_dir: str) -> dict:
    tested = [r for r in results if not r["skipped"]]
    skipped = [r for r in results if r["skipped"]]
    md = ["# Diagnóstico das falhas de busca e de raciocínio", "",
          f"Perfil `{cfg.get('profile')}` · gerador `{cfg['models']['generator']}`. Este teste não consta nos artigos: "
          "é um apoio à análise de falhas (`FALHAS.md`).", "",
          "Cada pergunta em que o método falhou por **busca** ou por **raciocínio** na rodada foi respondida de novo, "
          "com a etapa de geração do próprio método e os documentos já prontos:", "",
          "* **suporte** — só os parágrafos de suporte do gabarito;",
          "* **suporte+busca** — os documentos que o método recuperou, com os parágrafos de suporte que faltavam no "
          "lugar dos últimos documentos que não são de suporte.", "",
          f"**Critério de acerto:** {criterio_text(criterio, corte)}.", "",
          "**Métricas:** EM e F1 (avaliador do SQuAD/HotpotQA, as métricas dos dois artigos) e, como métrica "
          "adicional, o BLEU (Papineni et al., 2002), calculado como o `sentence_bleu` do sacreBLEU (suavização "
          "add-k com k = 1 e ordem efetiva). Todas sobre o texto normalizado, com o máximo entre as respostas de "
          "referência. Também como adicional, a similaridade de Levenshtein normalizada (Lev = 1 − d / maior "
          "comprimento, em caracteres, sobre o mesmo texto normalizado). As duas medem texto em comum, não correção: "
          "contra o julgamento manual de 161 pares resposta-gabarito do perfil mini, BLEU ≥ 0,5 concordou em 90,1%, "
          "Lev ≥ 0,7 em 91,3% e EM = 1 em 87,6%.", ""]
    for m in [k for k in METHODS if any(r["method"] == k for r in results)]:
        rs = [r for r in tested if r["method"] == m]
        md += [f"## {METHOD_NAMES[m]}", ""]
        rows = []
        for st in TARGETS:
            sub = [r for r in rs if r["stage"] == st]
            if not sub:
                continue
            rows.append([STAGE_NAMES[st], str(len(sub))] + [str(sum(r["label"] == lab for r in sub)) for lab in LABELS.values()])
        if rows:
            md += [md_table(["Etapa na rodada", "Perguntas"] + [lab[0].upper() + lab[1:] for lab in LABELS.values()], rows), ""]
        mrows = []
        for st in TARGETS:
            sub = [r for r in rs if r["stage"] == st]
            if not sub:
                continue
            mrows.append([STAGE_NAMES[st], "rodada original"] + [_mean(r["original"]["scores"][k] for r in sub) for k in METRICS])
            for cond in CONDITIONS:
                mrows.append([STAGE_NAMES[st], cond] + [_mean(r["conditions"][cond]["scores"][k] for r in sub) for k in METRICS])
        if mrows:
            md += ["Médias das métricas:", "", md_table(["Etapa", "Condição"] + [METRIC_NAMES[k] for k in METRICS], mrows,
                                                     "ll" + "r" * len(METRICS)), ""]
        crow = [[DATASET_NAMES.get(r["dataset"], r["dataset"]), f"`{r['id']}`", STAGE_NAMES[r["stage"]],
                 _clip(r["question"]), _clip(" | ".join(r["gold"]), 30), _clip(r["original"]["pred"], 30),
                 _clip(r["conditions"]["suporte"]["pred"], 30), _clip(r["conditions"]["suporte+busca"]["pred"], 30),
                 r["label"]] for r in rs]
        if crow:
            md += ["Perguntas:", "", md_table(["Dataset", "id", "Etapa", "Pergunta", "Gabarito", "Rodada", "suporte",
                                                "suporte+busca", "Rótulo"], crow, "l" * 9), ""]
    if skipped or notes:
        md += ["## Não testadas", ""] + [f"* {n}" for n in notes] + \
              [f"* {METHOD_NAMES[r['method']]} · {DATASET_NAMES.get(r['dataset'], r['dataset'])} · `{r['id']}`: "
               f"{r['skipped']}" for r in skipped] + [""]
    md += ["Para ver uma pergunta em detalhe (documentos, evidência, saída do modelo):",
           "`python -m ragbench diagnostico --profile <perfil> --metodo <método> --dataset <dataset> --id <id>`", ""]
    paths = {"md": os.path.join(out_dir, "DIAGNOSTICO.md"), "jsonl": os.path.join(out_dir, "diagnostico.jsonl"),
             "csv": os.path.join(out_dir, "diagnostico.csv")}
    with open(paths["md"], "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    write_jsonl(paths["jsonl"], results)
    with open(paths["csv"], "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["metodo", "dataset", "id", "etapa", "pergunta", "gabarito", "condicao", "resposta", "acertou"]
                   + list(METRICS) + ["rotulo"])
        for r in tested:
            base = [r["method"], r["dataset"], r["id"], r["stage"], r["question"], " | ".join(r["gold"])]
            w.writerow(base + ["rodada", r["original"]["pred"], ""] + [r["original"]["scores"][k] for k in METRICS] + [""])
            for cond, c in r["conditions"].items():
                w.writerow(base + [cond, c["pred"], "sim" if c["ok"] else "não"] + [c["scores"][k] for k in METRICS]
                           + [r["label"]])
    return paths


# ============================================================================ entrada
def run(cfg: dict, llm, methods: list[str], datasets: list[str] | None = None, targets: list[str] = TARGETS,
        ids: list[str] | None = None, criterio: str = "bleu", corte: float | None = None, limit: int | None = None) -> dict:
    log = get_logger()
    cases, notes = select_cases(cfg, methods, datasets, list(targets), ids)
    if limit:
        cases = cases[:limit]
    log.info(f"diagnóstico: {len(cases)} pergunta(s) a testar")
    results = []
    for i, case in enumerate(cases, 1):
        results.append(run_case(cfg, llm, case, criterio, corte))
        if i % 10 == 0 or i == len(cases):
            log.info(f"diagnóstico: {i}/{len(cases)}")
    out_dir = ensure_dir(os.path.join(cfg["paths"]["results"], "diagnostico"))
    if ids:
        write_json(os.path.join(out_dir, "pergunta_" + re.sub(r"[^\w.-]", "_", "_".join(ids))[:80] + ".json"), results)
        return {"results": results, "notes": notes, "paths": {}}
    return {"results": results, "notes": notes, "paths": write_report(cfg, results, notes, criterio, corte, out_dir)}
