"""Algoritmo 1 do DynaKRAG (inferência) + compressão terminal + variantes das ablações.

    1: inicializa s_0
    2: for t = 0..T−1:
    3:     constrói A(s_t)                                  (camada de validade)
    4:     remove ações que excedem o limite de retrieval/passos
    5:     a_t ← argmax_{a∈A(s_t)} [v̂θ(s_t,a) − λc(s_t,a)]   (+ modelo de continuação)
    6:     if a_t = stop_answer: break
    9:     executa a_t e atualiza s_{t+1}
   11: if compressão terminal: compress_answer_evidence
   14: gera a resposta com G

As regras de quando ``stop_answer`` e a sonda são válidos (``stop_rule``, ``max_probes``) ficam em
``validity.py`` e vêm do config pelo ambiente; cada linha de resultado leva a assinatura delas em ``control``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .state import STOP, support_recall
from .validity import retrieval_exhausted, valid_actions


@dataclass(frozen=True)
class Variant:
    name: str
    controller: str = "learned"            # learned | uniform
    compression: bool = True
    disabled: frozenset = field(default_factory=frozenset)
    retrieval_cap: int | None = None
    continuation: bool | None = None       # None = segue o config


VARIANTS = {
    "dynakrag": Variant("dynakrag"),
    "dynakrag_no_compression": Variant("dynakrag_no_compression", compression=False),
    "dynakrag_no_sufficiency": Variant("dynakrag_no_sufficiency", disabled=frozenset({"sufficiency_check"})),
    "dynakrag_uniform": Variant("dynakrag_uniform", controller="uniform"),
    "dynakrag_cap1": Variant("dynakrag_cap1", retrieval_cap=1),
    "dynakrag_cap2": Variant("dynakrag_cap2", retrieval_cap=2),
    "dynakrag_cap3": Variant("dynakrag_cap3", retrieval_cap=3),
}


def run_episode(env, controller, rec: dict, variant: Variant, max_steps: int) -> dict:
    s = env.init_state(rec)
    trace = []
    forced = False
    while True:
        A = valid_actions(s, env, max_steps, variant.retrieval_cap, variant.disabled)
        a, info = controller.choose(s, A, env)
        trace.append({"step": s.steps, "valid": A, "chosen": a, **info})
        if a == STOP:
            forced = A == [STOP]            # limite de passos/retrieval, ou nenhuma ação executável
            break
        s = env.step(s, a)
    docs = s.docs[: env.max_final]
    comp_raw = None
    comp_fallback = False
    n_llm = s.n_llm
    if variant.compression and docs:
        evidence, comp_raw = env.compress(s, docs)
        comp_fallback = evidence == env.passages(docs)      # o compressor não devolveu nenhuma linha com texto
        n_llm += 1
    else:
        evidence = env.passages(docs)
    pred, raw = env.answer(s.question, evidence)
    n_llm += 1
    return {
        "pred": pred, "raw": raw, "actions": list(s.history), "n_ret": s.n_ret, "steps": s.steps,
        "n_llm_actions": n_llm, "stopped_by": "cap" if forced else "controller",
        "exhausted": retrieval_exhausted(s, variant.retrieval_cap), "gaps": s.gaps_seen,
        "doc_titles": s.titles(), "doc_ids": [d["id"] for d in s.docs], "doc_via": [d["via"] for d in s.docs],
        "evidence": evidence,
        "compression_raw": comp_raw, "compression_fallback": comp_fallback, "trace": trace,
        "gate": getattr(controller, "gate", controller.name), "control": getattr(env, "control", None),
        "suff": s.suff, "n_checks": s.n_checks,
        "sr": support_recall(s.titles(), rec.get("supporting_titles") or []),  # diagnóstico (usa o gold)
    }
