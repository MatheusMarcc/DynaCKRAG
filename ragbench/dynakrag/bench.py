"""Suíte de benchmark do DynaKRAG (isolado): Tabela 1, Tabela 2, varredura de cap (Fig. 3 / Tab. 6),
composição das ações (Fig. 4 / Tab. 7), estrutura das perguntas (Fig. 5) e sonda (Tab. 8).

Saídas: ``<results>/dynakrag/<dataset>/<método>.jsonl`` (uma linha por pergunta).

Resultados e regras de controle. As variantes do DynaKRAG gravam em cada linha a assinatura das regras de
parada e da sonda (campo ``control``, ver ``state.control_signature``). Se o config mudar essas regras, os
resultados antigos não são reaproveitados nem apagados: vão para ``<results>/../logs/controle_anterior/`` e a
variante roda de novo. O controlador treinado segue a mesma lógica pelo nome do arquivo.
"""
from __future__ import annotations

import os
import re
import time
from functools import partial

from ..config import paper_datasets, retrieval_spec
from ..core.datasets import is_prepared, load_corpus, load_records
from ..core.io import ensure_dir, get_logger, move_file, read_jsonl
from ..core.jobs import Job
from ..core.retrieval import load_retriever
from ..core.runner import run_records
from . import baselines as B
from .controller import UniformController, controller_path, make_controller
from .env import DynaEnv
from .inference import VARIANTS, run_episode
from .state import control_signature, row_control, support_recall
from .training import train_controller

PAPER = "dynakrag"
MULTIHOP = {"hotpotqa", "2wiki", "musique", "toy_multihop"}


def results_dir(cfg: dict, dataset: str) -> str:
    return os.path.join(cfg["paths"]["results"], PAPER, dataset)


def result_path(cfg: dict, dataset: str, method: str) -> str:
    return os.path.join(results_dir(cfg, dataset), f"{method}.jsonl")


# ----------------------------------------------------------------------------- resultados de outras regras
def stale_control(cfg: dict, dataset: str, method: str) -> str | None:
    """Assinatura antiga, se o resultado desta variante do DynaKRAG foi gerado com outras regras de controle.

    Só as variantes do controlador dependem das regras; Fixed-K e os outros baselines nunca ficam desatualizados.
    """
    if method not in VARIANTS:
        return None
    sig = _file_control(result_path(cfg, dataset, method))
    return sig if sig and sig != control_signature(cfg) else None


_CONTROL_CACHE: dict = {}


def _file_control(path: str) -> str | None:
    """Assinatura de controle de um arquivo de resultado (a da primeira linha: o arquivo inteiro vem de uma rodada
    só, porque um arquivo com outras regras é guardado antes de a variante rodar de novo). Relido só se mudar."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    hit = _CONTROL_CACHE.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    rows = read_jsonl(path)
    sig = row_control(rows[0]) if rows else None
    if len(_CONTROL_CACHE) > 500:
        _CONTROL_CACHE.clear()
    _CONTROL_CACHE[path] = (mt, sig)
    return sig


def controller_missing(cfg: dict, dataset: str) -> bool:
    """True se ainda não há controlador treinado com as regras atuais (o job de treino precisa rodar)."""
    return not os.path.exists(controller_path(cfg, dataset))


def archive_stale(cfg: dict, dataset: str, method: str) -> str | None:
    """Guarda o resultado gerado com outras regras e libera o nome para a nova rodada. Devolve o destino."""
    old = stale_control(cfg, dataset, method)
    if not old:
        return None
    src = result_path(cfg, dataset, method)
    dst_dir = ensure_dir(os.path.join(cfg["paths"]["logs"], "controle_anterior", str(cfg.get("profile", "")), PAPER, dataset))
    tag = re.sub(r"[^a-z0-9]+", "-", old.lower()).strip("-")
    dst = os.path.join(dst_dir, f"{method}__{tag}.jsonl")
    if os.path.exists(dst):
        dst = os.path.join(dst_dir, f"{method}__{tag}__{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    move_file(src, dst)
    _CONTROL_CACHE.pop(src, None)
    get_logger().info(f"[DynaKRAG·{dataset}] {method}: o resultado anterior usava outras regras de controle ({old}); "
                      f"guardado em {dst} e a variante roda de novo com {control_signature(cfg)}")
    return dst


def make_env(cfg: dict, dataset: str, llm) -> DynaEnv:
    corpus = load_corpus(cfg, dataset)
    retr = load_retriever(corpus, cfg["paths"]["data"], dataset, retrieval_spec(cfg, PAPER))
    return DynaEnv(cfg, dataset, retr, llm)


def ensure_controller(cfg: dict, env: DynaEnv, force: bool = False) -> str:
    path = controller_path(cfg, env.dataset)
    if force or not os.path.exists(path):
        recs = load_records(cfg, env.dataset, "train", int(cfg["dynakrag"]["n_train"]))
        if not recs:
            raise RuntimeError(f"[{env.dataset}] sem perguntas de treino — rode prepare")
        train_controller(cfg, env, recs, force=force)
    return path


def method_names(cfg: dict) -> list[tuple[str, int]]:
    dk = cfg["dynakrag"]
    out = [(f"fixed_k{k}", 1) for k in dk["baselines"]["fixed_k_grid"]]
    out.append(("dynakrag", 1))
    out += [(f"dynakrag_{a}", 2) for a in dk.get("ablations", [])]
    for m in dk.get("methods", []):
        if m in ("fixed_k", "dynakrag"):
            continue
        if m == "corag":
            out += [("corag_s2", 3), ("corag_s3", 3)]
        else:
            out.append((m, 3))
    out += [(f"dynakrag_cap{c}", 3) for c in dk.get("cap_sweep", [])]
    return out


def _with_sr(rec: dict, res: dict) -> dict:
    if "sr" not in res:
        res["sr"] = support_recall(res.get("doc_titles", []), rec.get("supporting_titles") or [])
    return res


def build_fn(cfg: dict, env: DynaEnv, method: str):
    dk = cfg["dynakrag"]
    bc = dk["baselines"]
    max_steps = int(dk["max_steps"])
    if method.startswith("fixed_k"):
        k = int(method[len("fixed_k"):])
        return lambda rec: _with_sr(rec, B.fixed_k(env, rec, k))
    if method in VARIANTS:
        v = VARIANTS[method]
        if v.controller == "uniform":
            ctrl = UniformController(int(cfg.get("seed", 13)))
        else:
            ensure_controller(cfg, env)
            ctrl = make_controller(cfg, env.dataset, "learned", retrieval_cap=v.retrieval_cap, use_continuation=v.continuation)
        return lambda rec: run_episode(env, ctrl, rec, v, max_steps)
    if method == "ircot":
        return lambda rec: _with_sr(rec, B.ircot(env, rec, **bc["ircot"]))
    if method == "self_ask":
        return lambda rec: _with_sr(rec, B.self_ask(env, rec, **bc["self_ask"]))
    if method == "crag":
        return lambda rec: _with_sr(rec, B.crag(env, rec, **bc["crag"]))
    if method == "s2g_rag":
        return lambda rec: _with_sr(rec, B.s2g_rag(env, rec, **bc["s2g_rag"]))
    if method.startswith("corag_s"):
        steps = int(method[len("corag_s"):])
        return lambda rec: _with_sr(rec, B.corag(env, rec, k=bc["corag"]["k"], steps=steps))
    if method == "par2_rag":
        return lambda rec: _with_sr(rec, B.par2_rag(env, rec, **bc["par2_rag"]))
    if method == "adaptive_rag":
        ar = bc["adaptive_rag"]
        train = load_records(cfg, env.dataset, "train", int(ar["n_train"]))
        router = B.load_or_train_router(env, cfg, train, multihop=env.dataset in MULTIHOP)
        return lambda rec: _with_sr(rec, B.adaptive_rag(env, rec, router, k=ar["k"], ircot_kw=bc["ircot"]))
    raise ValueError(f"método desconhecido: {method}")


def run_method(cfg: dict, dataset: str, method: str, llm, limit: int | None = None) -> list[dict]:
    env = make_env(cfg, dataset, llm)
    recs = load_records(cfg, dataset, "eval", int(cfg["dynakrag"]["n_eval"]))
    if limit:
        recs = recs[:limit]
    fn = build_fn(cfg, env, method)
    archive_stale(cfg, dataset, method)
    out = result_path(cfg, dataset, method)
    return run_records(recs, fn, out, llm, desc=f"[DynaKRAG·{dataset}] {method}",
                       workers=int(cfg["llm"].get("workers", 1)))


def jobs(cfg: dict, llm, datasets: list[str] | None = None, methods: list[str] | None = None) -> list[Job]:
    log = get_logger()
    out = []
    for ds in datasets or paper_datasets(cfg, PAPER):
        if not is_prepared(cfg, ds):
            log.warning(f"[DynaKRAG] {ds} não está preparado — pulando (rode prepare)")
            continue
        # o treino do controlador vira um job próprio (tier 1), antes do método principal — só quando algum dos
        # métodos pedidos usa o controlador aprendido (pedir só o Fixed-K, ou só métodos de outro artigo, não treina nada)
        if not methods or any(m in VARIANTS and VARIANTS[m].controller == "learned" for m in methods):
            out.append(Job(PAPER, 1, ds, "train_controller",
                           partial(lambda d: ensure_controller(cfg, make_env(cfg, d, llm)), ds),
                           "rótulos da Eq. 1 + RF de valor + modelo de continuação"))
        for name, tier in method_names(cfg):
            if methods and name not in methods:
                continue
            out.append(Job(PAPER, tier, ds, name, partial(run_method, cfg, ds, name, llm)))
    return out
