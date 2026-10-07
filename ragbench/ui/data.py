"""O que o painel mostra — funções puras (config → dicionários prontos para JSON), sem HTTP.

O servidor só as embrulha em rotas; o ``--export`` chama as mesmas funções e embute o resultado no HTML.
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
from collections import defaultdict

from .. import analysis as A
from ..config import PAPERS, PROFILES, load_config, paper_datasets, shared_datasets
from ..core.datasets import DATASET_NAMES, dataset_dir
from ..core.io import read_json, read_jsonl
from ..core.text import norm_title, normalize_answer

DOC_CHARS = 1600          # corte do texto de um parágrafo no painel
RAW_CHARS = 6000          # corte de uma saída do modelo
ACTION_PT = {"retrieve_more": "buscar mais (próximos do ranking)", "gap_query": "buscar a lacuna apontada pela sonda",
             "rewrite_query": "reescrever a pergunta e buscar", "bridge_entity_expand": "expandir a entidade-ponte",
             "sufficiency_check": "checar se a evidência é suficiente", "stop_answer": "parar e responder"}


def clip(s, n: int = RAW_CHARS) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + f"\n… (cortado: mais {len(s) - n} caracteres)"


# ============================================================================ perfis e estado
def profiles(root: str, overrides: list[str] | None = None) -> list[dict]:
    """Os perfis e o que cada um já tem de resultado."""
    out = []
    for name in PROFILES:
        try:
            cfg = load_config(name, overrides or [], root)
        except SystemExit:
            continue
        files = A.result_files(cfg)
        paths = [p for ms in files.values() for p in ms.values()]
        out.append({"name": name, "n_files": len(paths), "datasets": A.dataset_order(files),
                    "mtime": max((os.path.getmtime(p) for p in paths), default=0),
                    "n_eval": {p: cfg[p]["n_eval"] for p in PAPERS}, "backend": cfg["llm"]["backend"]})
    return out


def latest_profile(root: str, overrides: list[str] | None = None, default: str = "quick") -> str:
    """O perfil com resultado mais recente (o 'smoke' só se for o único)."""
    ps = [p for p in profiles(root, overrides) if p["n_files"]]
    real = [p for p in ps if p["name"] != "smoke"] or ps
    return max(real, key=lambda p: p["mtime"])["name"] if real else default


def _dataset_info(cfg: dict, ds: str, files: dict) -> dict:
    d = dataset_dir(cfg, ds)
    meta = read_json(os.path.join(d, "meta.json"), {}) or {}
    idx = sorted(os.path.basename(p) for p in glob.glob(os.path.join(d, "index", "*")) if not p.endswith(".json"))
    return {"key": ds, "name": DATASET_NAMES.get(ds, ds), "prepared": bool(meta) and os.path.exists(os.path.join(d, "eval.jsonl")),
            "meta": meta, "indexes": idx, "n_methods": len(files.get(ds, {})),
            "papers": [p for p in PAPERS if ds in paper_datasets(cfg, p)],
            "compare": ds in shared_datasets(cfg)}


def state(cfg: dict) -> dict:
    files = A.result_files(cfg)
    known = set(files) | {ds for p in PAPERS for ds in paper_datasets(cfg, p)}
    cmp_cfg = cfg.get("compare") or {}
    res = cfg["paths"]["results"]
    eb = os.path.join(res, "ragcritic", "errorbench")
    return {
        "profile": cfg["profile"], "root": cfg["root"], "results": res, "backend": cfg["llm"]["backend"],
        "datasets": [_dataset_info(cfg, ds, files) for ds in A.dataset_order(known)],
        "compare": {"datasets": shared_datasets(cfg), "metric": str(cmp_cfg.get("fail_metric", "em")),
                    "thr": float(cmp_cfg.get("fail_threshold", 0.5)),
                    "default_methods": list(cmp_cfg.get("default_methods") or [])},
        "models": dict(cfg["models"]), "n_eval": {p: cfg[p]["n_eval"] for p in PAPERS},
        "retrieval": {p: dict(cfg["retrieval"][p]) for p in PAPERS},
        "jobs": {k: v for k, v in (read_json(os.path.join(res, "status.json"), {}) or {}).items()
                 if k.split("/", 1)[0] in PAPERS},
        "errorbench": sorted(os.path.basename(p)[: -len("__original.jsonl")]
                             for p in glob.glob(os.path.join(eb, "*__original.jsonl"))),
        "families": A.FAMILY_LABEL, "papers": A.PAPER_LABEL, "stages": A.STAGE_LABEL,
    }


_ROWS: dict = {}
_MATRIX: dict = {}


def rows_by_id(path: str) -> dict:
    """{id: linha} de um JSONL de resultados, relido só quando o arquivo muda."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return {}
    hit = _ROWS.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    data = {r["id"]: r for r in read_jsonl(path) if "id" in r}
    if len(_ROWS) > 300:
        _ROWS.clear()
    _ROWS[path] = (mt, data)
    return data


def matrix(cfg: dict, dataset: str) -> dict:
    """A matriz testes × métodos de um dataset (em cache enquanto os arquivos não mudam)."""
    files = A.result_files(cfg).get(dataset, {})
    sig = (cfg["paths"]["results"], dataset, tuple((p, os.path.getmtime(p)) for p in files.values()))
    hit = _MATRIX.get((sig[0], dataset))
    if hit and hit[0] == sig:
        return hit[1]
    m = A.load_matrix(cfg, dataset)
    m["defaults"] = A.default_methods(cfg, set(m["cells"]))
    _MATRIX[(sig[0], dataset)] = (sig, m)
    return m


# ============================================================================ plano e estimativa de uma rodada
# Peso de uma chamada em cada artigo, em relação a uma chamada simples (prompt curto, resposta curta):
# o crítico e o planner escrevem saídas longas.
CALL_WEIGHT = {"ragcritic": 1.5, "dynakrag": 1.0}


def seconds_per_call(cfg: dict) -> tuple[float, bool]:
    """Segundos de relógio por chamada simples ao LLM, medidos nos resultados que já existem (ou 3 s, sem medida).

    Usa as bases de uma chamada só (Standard RAG, Fixed-K): o tempo inclui a busca e a carga do modelo.
    """
    def measure(c: dict) -> tuple[float, float]:
        secs = calls = 0.0
        for ms in A.result_files(c).values():
            for key, path in ms.items():
                if not (key.endswith("/standard_rag") or "/fixed_k" in key):
                    continue
                for r in read_jsonl(path)[:100]:
                    n, s = r.get("llm_calls") or 0, r.get("secs") or r.get("llm_seconds") or 0
                    if n and s:
                        secs += s
                        calls += n
        return secs, calls

    secs, calls = measure(cfg)
    if calls < 20:                                  # este perfil ainda não rodou: vale a medida de outro perfil
        for name in PROFILES:
            if name in (cfg.get("profile"), "smoke"):
                continue
            try:
                s2, c2 = measure(load_config(name, [], cfg["root"]))
            except SystemExit:
                continue
            if c2 >= 20:
                secs, calls = s2, c2
                break
    return (secs / calls, True) if calls >= 20 and secs > 0 else (3.0, False)


def plan(cfg: dict, datasets=None, papers=None, tiers=None) -> dict:
    from ..doctor import estimate_job, planned_jobs
    from ..dynakrag import bench as dkb

    status = read_json(os.path.join(cfg["paths"]["results"], "status.json"), {}) or {}
    files = A.result_files(cfg)
    per, measured = seconds_per_call(cfg)
    jobs = []
    for j in planned_jobs(cfg, datasets or None, papers or None, tiers or None):
        st = (status.get(j["key"]) or {}).get("state")
        has_file = f"{j['paper']}/{j['name']}" in files.get(j["dataset"], {})
        # pronto = o agendador registrou que terminou (ou há resultado sem registro, ex.: pasta copiada de outra máquina);
        # "running" que sobrou de uma rodada interrompida ainda falta rodar (a rodada retoma de onde parou)
        state = "done" if (st == "done" or (st is None and has_file)) else ("failed" if st == "failed" else "pending")
        # DynaKRAG: resultado (ou controlador) gerado com outras regras de parada/sonda não conta como pronto — roda de novo
        stale = False
        if j["paper"] == "dynakrag" and state == "done":
            if j["name"] == "train_controller":
                stale = dkb.controller_missing(cfg, j["dataset"])
            else:
                stale = bool(dkb.stale_control(cfg, j["dataset"], j["name"]))
            if stale:
                state = "pending"
        calls = estimate_job(cfg, j["paper"], j["dataset"], j["name"])
        prepared = j["dataset"] == "errorbench" or os.path.exists(os.path.join(dataset_dir(cfg, j["dataset"]), "eval.jsonl"))
        jobs.append({**j, "state": state, "calls": int(calls), "prepared": prepared, "stale": stale})
    todo = [j for j in jobs if j["state"] != "done"]
    calls = sum(j["calls"] for j in todo)
    secs = sum(j["calls"] * CALL_WEIGHT.get(j["paper"], 1.0) for j in todo) * per
    return {"jobs": jobs, "n": len(jobs), "n_todo": len(todo), "calls": calls, "secs_per_call": round(per, 2),
            "measured": measured, "hours": round(secs / 3600, 2), "n_stale": sum(1 for j in jobs if j["stale"]),
            "missing_data": sorted({j["dataset"] for j in jobs if not j["prepared"]})}


# ============================================================================ documentos de um teste
class DocReg:
    """Documentos citados por um teste, sem repetição (os métodos recuperam quase sempre os mesmos)."""

    def __init__(self, golds: list[str], support: list[str]):
        self.docs: dict = {}
        self._keys: dict = {}
        self.gold = [g for g in (normalize_answer(x) for x in golds or []) if g and g not in ("yes", "no", "noanswer")]
        self.sup = {norm_title(t) for t in support or [] if t}

    def add(self, d: dict) -> str:
        ident = d.get("id") or hashlib.md5(f"{d.get('title')}|{(d.get('text') or '')[:200]}".encode()).hexdigest()[:10]
        key = self._keys.get(ident)
        if key is None:
            key = self._keys[ident] = f"d{len(self.docs)}"
            text = d.get("text") or ""
            full = normalize_answer(f"{d.get('title', '')} {text}")
            self.docs[key] = {"id": d.get("id"), "title": d.get("title", ""), "text": clip(text, DOC_CHARS),
                              "sup": norm_title(d.get("title", "")) in self.sup,
                              "ans": any(g in full for g in self.gold)}
        return key

    def add_all(self, docs: list[dict]) -> list[str]:
        return list(dict.fromkeys(self.add(d) for d in docs))


def _step(kind: str, title: str, text=None, docs=None) -> dict:
    s = {"k": kind, "t": title}
    if text not in (None, ""):
        s["x"] = clip(text)
    if docs:
        s["d"] = docs
    return s


# ---------------------------------------------------------------------------- RAG-Critic
def _critic_key(cfg: dict, name: str) -> str | None:
    from ..ragcritic.workflow import variants

    v = variants(cfg).get(name)
    if v is None:
        return None
    indent = 16 if str(cfg["ragcritic"].get("critic_template", "indent20")).endswith("16") else 20
    return f"{v.critic}{indent}"


def _steps_ragcritic(cfg: dict, base: str, name: str, row: dict, reg: DocReg, docs: list[str], stage) -> list[dict]:
    steps = []
    if name == "standard_rag":
        steps.append(_step("busca", f"Busca com a pergunta (top-{len(docs) or cfg['ragcritic']['top_k']})", docs=docs))
        steps.append(_step("resposta", "Geração (Prompt A do artigo)", row.get("raw")))
        return steps
    if "trace" in row:
        steps.append(_step("resposta", "Resposta inicial (Standard RAG)", row.get("initial_pred") or "(vazia)"))
        ckey = _critic_key(cfg, name)
        for tr in row.get("trace") or []:
            it = tr.get("it", 1)
            crit = stage(f"critic__{ckey}__it{it}") if ckey else {}
            lines = [f"Julgamento: {tr.get('judgement')}"]
            for t in ("tag1", "tag2"):
                if tr.get(t):
                    lines.append(f"{t}: {', '.join(map(str, tr[t]))}")
            if crit.get("tag3"):
                lines.append(f"tag3: {', '.join(map(str, crit['tag3']))}")
            if crit.get("analysis"):
                lines.append(f"Análise do crítico: {crit['analysis']}")
            if crit.get("raw"):
                lines.append(f"\nSaída bruta:\n{crit['raw']}")
            steps.append(_step("critico", f"Crítico, rodada {it}" + (f" ({crit['critic']})" if crit.get("critic") else ""),
                               "\n".join(lines)))
            if "program" in tr:
                steps.append(_step("plano", "Programa de correção escrito pelo planner", tr.get("program") or "(vazio)"))
                ex = [f"Execução: {'ok' if tr.get('exec_ok') else 'FALHOU'}"]
                if tr.get("exec_error"):
                    ex.append(f"Erro: {tr['exec_error']}")
                if tr.get("calls"):
                    ex.append("Funções chamadas: " + " → ".join(map(str, tr["calls"])))
                if tr.get("queries"):
                    ex.append("Consultas feitas:\n" + "\n".join(f"  • {q}" for q in tr["queries"]))
                ex.append(f"Resposta antes: {tr.get('prev_pred') or '(vazia)'}\nResposta depois: {tr.get('new_pred') or '(vazia)'}")
                steps.append(_step("execucao", "Execução do programa", "\n".join(ex)))
        steps.append(_step("resposta", "Saída final do gerador", row.get("raw")))
        return steps
    # baselines do artigo
    steps.append(_step("resposta", "Resposta inicial (Standard RAG)", (stage("standard_rag") or {}).get("pred")))
    for i, fb in enumerate(row.get("feedbacks") or [], 1):
        steps.append(_step("critico", f"Feedback do próprio modelo, rodada {i}", fb))
    if row.get("flare_mode"):
        txt = f"Modo: {row['flare_mode']}"
        for r in row.get("rounds") or []:
            txt += f"\nNova busca ({r.get('n_low')} tokens de baixa confiança): {r.get('query')}"
        steps.append(_step("busca", "FLARE: busca ativa", txt))
    if row.get("meta_action"):
        txt = (f"Resposta só com o conhecimento interno: {row.get('internal')}\n"
               f"Semelhança com a resposta do RAG: {row.get('similarity')}\nDecisão: {row['meta_action']}")
        if row.get("evaluation"):
            txt += f"\nAvaliação: {row['evaluation']}"
        if row.get("rewritten_query"):
            txt += f"\nNova consulta: {row['rewritten_query']}"
        steps.append(_step("critico", "MetaRAG: monitorar, avaliar, planejar", txt))
    steps.append(_step("resposta", "Saída final", row.get("raw")))
    return steps


# ---------------------------------------------------------------------------- DynaKRAG
_DK_EXTRA = [("cot", "Cadeia de raciocínio (IRCoT)"), ("transcript", "Transcrição (Self-Ask)"), ("chain", "Cadeia de sub-perguntas (CoRAG)"),
             ("rounds", "Rodadas de suficiência (S2G-RAG)"), ("subquestions", "Sub-perguntas (PAR²-RAG)"),
             ("refine", "Refino em profundidade (PAR²-RAG)"), ("crag_action", "Decisão do avaliador (CRAG)"),
             ("labels", "Rótulos dos documentos (CRAG)"), ("route", "Rota escolhida (Adaptive-RAG: A sem busca, B um passo, C multi-passo)"),
             ("known_facts", "Fatos conhecidos (S2G-RAG)")]


def _fmt_value(v) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "\n".join(f"  • {_fmt_value(x)}" if not isinstance(x, (list, tuple)) else "  • " + " → ".join(map(str, x)) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k}: {x}" for k, x in v.items())
    return str(v)


def _steps_dynakrag(name: str, row: dict, reg: DocReg, docs: list[str], by_title: dict) -> list[dict]:
    steps = []
    if "trace" in row and "actions" in row:
        via = defaultdict(list)              # ação → documentos que ela trouxe
        for t, a in zip(row.get("doc_titles") or [], row.get("doc_via") or []):
            if t in by_title:
                via[a].append(by_title[t])
        shown = set()
        for tr in row.get("trace") or []:
            chosen = tr.get("chosen")
            lines = ["Ações válidas: " + ", ".join(tr.get("valid") or [])]
            if tr.get("scores"):
                lines.append("Valor previsto pelo controlador: " + ", ".join(
                    f"{a} = {v:+.3f}" for a, v in sorted(tr["scores"].items(), key=lambda kv: -kv[1])))
            if "p_continue" in tr:
                lines.append(f"Chance de que continuar ajude: {tr['p_continue']}"
                             + (" → a parada foi adiada" if tr.get("stop_suppressed") else ""))
            if tr.get("forced"):
                lines.append("Parada forçada: não havia outra ação válida (limite de passos ou de buscas).")
            d = None
            if chosen in via and chosen not in shown:      # os documentos aparecem na 1ª vez que a ação roda
                d = via[chosen]
                shown.add(chosen)
            steps.append(_step("acao", f"Passo {int(tr.get('step', 0)) + 1}: {ACTION_PT.get(chosen, chosen)}", "\n".join(lines), d))
        if row.get("gaps"):
            steps.append(_step("critico", "Lacunas apontadas pela sonda de suficiência", _fmt_value(row["gaps"])))
        if row.get("compression_raw") or row.get("evidence"):
            steps.append(_step("evidencia", "Evidência entregue ao gerador" + (" (comprimida)" if row.get("compression_raw") else ""),
                               row.get("evidence")))
    else:
        steps.append(_step("busca", f"Documentos reunidos ({len(docs)})", docs=docs))
        for key, title in _DK_EXTRA:
            if row.get(key) not in (None, "", [], {}):
                steps.append(_step("raciocinio", title, _fmt_value(row[key])))
    steps.append(_step("resposta", "Saída do gerador", row.get("raw")))
    return steps


# ============================================================================ um teste, todos os métodos
def test_detail(cfg: dict, dataset: str, qid: str, methods: list[str] | None = None) -> dict | None:
    files = A.result_files(cfg).get(dataset, {})
    recs = A.load_eval(cfg, dataset)
    corpus = A.corpus_index(cfg, dataset)
    res = cfg["paths"]["results"]

    def stage_row(paper: str, fname: str) -> dict:
        return rows_by_id(os.path.join(res, paper, dataset, fname + ".jsonl")).get(qid) or {}

    rows = {}
    for key, path in files.items():
        if methods and key not in methods:
            continue
        row = rows_by_id(path).get(qid)
        if row and not row.get("error"):
            rows[key] = row
    rec = recs.get(qid) or next(iter(rows.values()), None)
    if rec is None:
        return None
    golds, sup = list(rec.get("answers") or []), list(rec.get("supporting_titles") or [])
    reg = DocReg(golds, sup)
    support_docs = []
    if corpus:
        for t in sup:
            support_docs += reg.add_all(corpus["by_title"].get(norm_title(t), [])[:2])
    std = stage_row("ragcritic", "standard_rag")
    out_methods = {}
    for key, row in rows.items():
        paper, _, name = key.partition("/")
        fb = std if (paper == "ragcritic" and name != "standard_rag") else None
        docs = A.row_docs(row, corpus, fb)
        keys = reg.add_all(docs)
        by_title = {}
        for d, k in zip(docs, keys):
            by_title.setdefault(d.get("title"), k)
        cell = A.make_cell(row, recs.get(qid), corpus, fb)
        try:
            if paper == "ragcritic":
                steps = _steps_ragcritic(cfg, key, name, row, reg, keys, lambda f: stage_row("ragcritic", f))
            else:
                steps = _steps_dynakrag(name, row, reg, keys, by_title)
        except Exception as e:  # noqa: BLE001 — um formato inesperado não derruba a página do teste
            steps = [_step("nota", "Não foi possível montar o passo a passo", f"{type(e).__name__}: {e}"),
                     _step("resposta", "Saída do modelo", row.get("raw"))]
        meta = {"tokens": row.get("tokens"), "chamadas ao LLM": row.get("llm_calls"), "buscas": row.get("n_ret"),
                "segundos de LLM": row.get("llm_seconds")}
        out_methods[key] = {**cell, "docs": keys, "steps": steps, "meta": {k: v for k, v in meta.items() if v is not None},
                            "info": A.method_info(key)}
    return {"dataset": dataset, "id": qid, "question": rec.get("question", ""), "answers": golds, "type": rec.get("type") or "",
            "hops": rec.get("hops"), "support": sup, "support_docs": support_docs, "docs": reg.docs,
            "methods": out_methods, "has_corpus": corpus is not None}


# ============================================================================ aba Dados
def questions(cfg: dict, dataset: str, split: str = "eval", q: str = "", offset: int = 0, limit: int = 50) -> dict:
    split = split if split in ("eval", "train") else "eval"
    recs = read_jsonl(os.path.join(dataset_dir(cfg, dataset), f"{split}.jsonl"))
    used = {p: int(cfg[p]["n_eval"] if split == "eval" else cfg[p].get("n_train", 0) or 0) for p in PAPERS}
    ql = (q or "").strip().lower()
    items = []
    for i, r in enumerate(recs):
        if ql and ql not in f"{r.get('id')} {r.get('question')} {' '.join(map(str, r.get('answers') or []))} {r.get('type')}".lower():
            continue
        items.append({"n": i + 1, "id": r.get("id"), "q": r.get("question", ""), "gold": list(r.get("answers") or []),
                      "type": r.get("type") or "", "hops": r.get("hops"), "sup": list(r.get("supporting_titles") or []),
                      "used": i < max(used.values())})
    return {"dataset": dataset, "split": split, "total": len(recs), "matched": len(items), "offset": offset,
            "items": items[offset:offset + limit], "used": used,
            "meta": read_json(os.path.join(dataset_dir(cfg, dataset), "meta.json"), {}) or {}}


def corpus_search(cfg: dict, dataset: str, q: str = "", offset: int = 0, limit: int = 30) -> dict:
    corpus = A.corpus_index(cfg, dataset)
    if corpus is None:
        return {"dataset": dataset, "total": 0, "matched": 0, "offset": 0, "items": [], "missing": True}
    sup = {}
    for r in A.load_eval(cfg, dataset).values():
        for t in r.get("supporting_titles") or []:
            sup[norm_title(t)] = sup.get(norm_title(t), 0) + 1
    words = [w for w in re.split(r"\s+", (q or "").lower().strip()) if w]
    docs = list(corpus["by_id"].values())
    if words:
        docs = [d for d in docs if all(w in f"{d.get('title', '')} {d.get('text', '')}".lower() for w in words)]
    items = [{"id": d["id"], "title": d.get("title", ""), "text": clip(d.get("text", ""), 900),
              "sup": sup.get(norm_title(d.get("title", "")), 0)} for d in docs[offset:offset + limit]]
    return {"dataset": dataset, "total": corpus["n"], "matched": len(docs), "offset": offset, "items": items}


_EB_CACHE: dict = {}


def _errorbench_items(cfg: dict) -> dict:
    from ..ragcritic.errorbench import bench_dir, load_items

    path = os.path.join(bench_dir(cfg), "critic_model_test.jsonl")
    if not os.path.exists(path):
        return {}
    key = (path, os.path.getmtime(path))
    if key not in _EB_CACHE:
        _EB_CACHE.clear()
        _EB_CACHE[key] = {x["id"]: x for x in load_items(path)}
    return _EB_CACHE[key]


def errorbench(cfg: dict, tag: str = "", offset: int = 0, limit: int = 20, label: str = "", show: str = "") -> dict:
    """Exemplos do RAG-Error Bench com o julgamento do crítico (prompt original e com o recuo trocado)."""
    from ..ragcritic import prompts as P

    out_dir = os.path.join(cfg["paths"]["results"], "ragcritic", "errorbench")
    tags = sorted(os.path.basename(p)[: -len("__original.jsonl")] for p in glob.glob(os.path.join(out_dir, "*__original.jsonl")))
    tag = tag if tag in tags else (tags[0] if tags else "")
    orig = {r["id"]: r for r in read_jsonl(os.path.join(out_dir, f"{tag}__original.jsonl"))} if tag else {}
    swap = {r["id"]: r for r in read_jsonl(os.path.join(out_dir, f"{tag}__swapped.jsonl"))} if tag else {}
    items_all = _errorbench_items(cfg)
    ids = [i for i in orig if (not label or orig[i].get("label") == label)]
    if show == "errou":
        ids = [i for i in ids if orig[i].get("judgement") != orig[i].get("label")]
    elif show == "mudou":
        ids = [i for i in ids if i in swap and swap[i].get("judgement") != orig[i].get("judgement")]
    items = []
    for i in ids[offset:offset + limit]:
        x = items_all.get(i) or {}
        parsed = {}
        if x.get("instruction"):
            try:
                parsed = P.parse_critic_instruction(x["instruction"])
            except Exception:  # noqa: BLE001
                parsed = {}

        def res(r):
            return None if not r else {"judgement": r.get("judgement"), "tag1": r.get("tag1"), "tag2": r.get("tag2"),
                                       "indent": r.get("indent"), "ok": r.get("judgement") == r.get("label"),
                                       "raw": clip(r.get("raw"), 1500)}

        gold = x.get("gold") or {}
        items.append({"id": i, "label": orig[i].get("label"), "question": parsed.get("question", ""),
                      "prediction": clip(parsed.get("prediction", ""), 1200),
                      "passages": [clip(p, 900) for p in parsed.get("passages", [])],
                      "gold": {k: gold.get(k) for k in ("Judgement", "Error_analysis", "tag1", "tag2", "tag3") if gold.get(k)},
                      "original": res(orig.get(i)), "swapped": res(swap.get(i))})
    n = len(orig)
    summary = {"n": n, "ok_original": sum(r.get("judgement") == r.get("label") for r in orig.values()),
               "n_swapped": len(swap), "ok_swapped": sum(r.get("judgement") == r.get("label") for r in swap.values()),
               "flipped": sum(1 for i, r in swap.items() if i in orig and r.get("judgement") != orig[i].get("judgement")),
               "by_indent": {}}
    for r in list(orig.values()) + list(swap.values()):
        k = f"{r.get('indent')}→{r.get('judgement')}"
        summary["by_indent"][k] = summary["by_indent"].get(k, 0) + 1
    return {"tags": tags, "tag": tag, "matched": len(ids), "offset": offset, "items": items, "summary": summary,
            "has_items": bool(items_all), "artifact": read_json(os.path.join(out_dir, "format_artifact.json"), {}) or {}}


# ============================================================================ relatórios
def reports(cfg: dict) -> list[dict]:
    res = cfg["paths"]["results"]
    names = [("REPORT.md", "Índice e estado dos jobs"), ("TABELAS.md", "Tabelas no padrão dos artigos"),
             ("FALHAS.md", "Análise de falhas"),
             ("COMPARACAO.md", "Comparação com os artigos"), ("ragcritic/REPORT.md", "RAG-Critic isolado"),
             ("dynakrag/REPORT.md", "DynaKRAG isolado")]
    return [{"path": p, "title": t, "mtime": os.path.getmtime(os.path.join(res, p))}
            for p, t in names if os.path.exists(os.path.join(res, p))]


def report_text(cfg: dict, path: str) -> str | None:
    if path not in {r["path"] for r in reports(cfg)}:
        return None
    with open(os.path.join(cfg["paths"]["results"], path), encoding="utf-8") as f:
        return f.read()


def safe_result_file(cfg: dict, rel: str) -> str | None:
    """Caminho absoluto de uma figura dos relatórios — só .png dentro da pasta de resultados."""
    res = os.path.realpath(cfg["paths"]["results"])
    full = os.path.realpath(os.path.join(res, rel))
    if not full.lower().endswith(".png") or os.path.commonpath([res, full]) != res or not os.path.isfile(full):
        return None
    return full
