"""Comparação com os artigos: vereditos, Δ pareado e arquivos gerados."""
import csv
import json
import os
import random

from ragbench.compare import build, dynakrag_claims, format_audit, ragcritic_claims, verdict
from ragbench.config import load_config


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_verdict_rules():
    assert verdict(5.0, 3.0) == "mesmo sinal"
    assert verdict(-5.0, 3.0) == "sinal oposto"
    assert verdict(0.0, 3.0) == "sem diferença"
    assert verdict(float("nan"), 3.0) == "falta rodar"
    assert verdict(2.0, None) == "sem referência"
    assert verdict(-434.0, -434.2) == "mesmo sinal"      # tokens: negativo = economiza, igual ao artigo


def test_build_with_synthetic_results(tmp_path):
    cfg = load_config("smoke", ["dynakrag.datasets=[hotpotqa]", "ragcritic.datasets=[nq]"],
                      root=str(tmp_path))
    res = cfg["paths"]["results"]
    rng = random.Random(0)
    ids = [f"q{i}" for i in range(60)]
    good = [{"id": i, "f1": 1.0, "em": 1.0, "tokens": 900, "llm_calls": 4, "n_ret": 2, "sr": 0.9} for i in ids]
    bad = [{"id": i, "f1": float(rng.random() < 0.3), "em": 0.0, "tokens": 1500, "llm_calls": 5, "n_ret": 3, "sr": 0.5}
           for i in ids]
    _write(os.path.join(res, "dynakrag", "hotpotqa", "dynakrag.jsonl"), good)
    _write(os.path.join(res, "dynakrag", "hotpotqa", "fixed_k6.jsonl"), bad)
    _write(os.path.join(res, "dynakrag", "hotpotqa", "s2g_rag.jsonl"), bad)
    _write(os.path.join(res, "ragcritic", "nq", "ragcritic.jsonl"), bad)      # RAG-Critic PIOR que o Standard RAG
    _write(os.path.join(res, "ragcritic", "nq", "standard_rag.jsonl"), good)
    # auditoria: o crítico acerta tudo no original e erra os 'Correct' quando o recuo é trocado
    eb = os.path.join(res, "ragcritic", "errorbench")
    orig = [{"id": f"b{i}", "label": lab, "judgement": lab} for i, lab in enumerate(["Correct"] * 30 + ["Error"] * 30)]
    swap = [{"id": r["id"], "label": r["label"], "judgement": "Error"} for r in orig]
    _write(os.path.join(eb, "rag-critic-3b__original.jsonl"), orig)
    _write(os.path.join(eb, "rag-critic-3b__swapped.jsonl"), swap)

    dk = {c["claim"]: c for c in dynakrag_claims(cfg)}
    assert dk["DynaKRAG supera o Fixed-K (melhor K)"]["verdict"] == "mesmo sinal"
    assert "p" not in dk["DynaKRAG supera o Fixed-K (melhor K)"]
    assert dk["DynaKRAG supera o Fixed-K (melhor K)"]["b"] == "Fixed-K (K=6)"
    assert dk["DynaKRAG gasta menos tokens que o S2G-RAG"]["verdict"] == "mesmo sinal"
    assert dk["a sonda de suficiência ajuda"]["verdict"] == "falta rodar"
    rc = {c["claim"]: c for c in ragcritic_claims(cfg)}
    assert rc["RAG-Critic supera o Standard RAG"]["verdict"] == "sinal oposto"
    assert rc["uma 2ª iteração ainda ajuda"]["verdict"] == "falta rodar"

    audit = {a["classe"]: a for a in format_audit(cfg)}
    assert audit["Correct"]["delta"] == 100.0 and audit["Correct"]["decisoes_mudaram"] == 100.0 and "p" not in audit["Correct"]
    assert audit["Error"]["delta"] == 0.0

    path = build(cfg)
    md = open(path, encoding="utf-8").read()
    assert "**mesmo sinal**" in md and "**sinal oposto**" in md and "o crítico depende do formato" in md
    assert "p-valor" not in md and "IC 95" not in md and "p < 0,05" not in md and "nosso" not in md
    with open(os.path.join(res, "metrics.csv"), encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f, delimiter=";"))
    by = {(r["artigo"], r["dataset"], r["metodo"]): r for r in rows}
    assert by[("dynakrag", "hotpotqa", "dynakrag")]["F1"] == "1"
    assert by[("dynakrag", "hotpotqa", "dynakrag")]["artigo_F1"] == "0,5998"
    assert not any("IC95" in k for k in rows[0])
    assert {r["artigo"] for r in rows} == {"dynakrag", "ragcritic"}
    # resultados de um artigo que saiu do projeto (pasta antiga em ragbench_results) não entram em nada
    _write(os.path.join(res, "doctorrag", "hotpotqa", "react.jsonl"), good)
    build(cfg)
    with open(os.path.join(res, "metrics.csv"), encoding="utf-8-sig") as f:
        assert {r["artigo"] for r in csv.DictReader(f, delimiter=";")} == {"dynakrag", "ragcritic"}
    with open(os.path.join(res, "comparacao.csv"), encoding="utf-8-sig") as f:
        assert {r["artigo"] for r in csv.DictReader(f, delimiter=";")} == {"DynaKRAG", "RAG-Critic"}
