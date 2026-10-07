"""``python -m ragbench doctor``: confere ambiente, Ollama, modelos, % na GPU, logprobs e estima o tempo."""
from __future__ import annotations

import importlib
import platform
import sys
import time

from .core.llm import render_chatml

PACKAGES = [("numpy", True), ("scipy", True), ("sklearn", True), ("yaml", True), ("httpx", True), ("json_repair", True),
            ("tqdm", True), ("matplotlib", True), ("datasets", True), ("huggingface_hub", True),
            ("sentence_transformers", False), ("tokenizers", False)]

_SAMPLE_DOCS = "\n".join(
    f"Passage: \"Document {i}\"\n" + ("The quick brown fox jumps over the lazy dog near the river bank in the old town. " * 9)
    for i in range(5))


def _models_needed(cfg: dict) -> dict:
    m = cfg["models"]
    need = {r: m[r] for r in ("generator", "critic", "planner") if r in m}
    for z in cfg["ragcritic"]["errorbench"].get("zeroshot_models", []):
        need[f"zero-shot ({z})"] = m.get(z, z)
    return need


# ============================================================================ plano e estimativa
# chamadas ao LLM por pergunta, em média (GROSSEIRO; já descontando o que os estágios e o cache compartilham)
_RC_CALLS = {"standard_rag": 1, "ragcritic": 3, "ragcritic_n2": 2, "ragcritic_zeroshot_critic": 4,
             "ragcritic_rule_planning": 2.5, "self_refine": 3, "flare": 2, "metarag": 3}
_DK_CALLS = {"dynakrag": 4, "dynakrag_no_compression": 3, "dynakrag_no_sufficiency": 3, "dynakrag_uniform": 3,
             "ircot": 5, "self_ask": 7, "crag": 4, "adaptive_rag": 1.5, "s2g_rag": 3.5, "corag_s2": 5, "corag_s3": 7,
             "par2_rag": 5.5}


def planned_jobs(cfg: dict, datasets=None, papers=None, tiers=None) -> list[dict]:
    """Os jobs que ``bench`` rodaria (sem checar se os dados estão preparados): [{paper, tier, dataset, name, key}]."""
    from .config import PAPERS, paper_datasets
    from .dynakrag.bench import method_names as dk_names
    from .ragcritic.bench import method_names as rc_names

    out = []

    def add(paper, tier, ds, name):
        if (not papers or paper in papers) and (not tiers or tier in tiers):
            out.append({"paper": paper, "tier": tier, "dataset": ds, "name": name, "key": f"{paper}/{ds}/{name}"})

    add("ragcritic", 1, "errorbench", "rag-critic-3b")
    for m in cfg["ragcritic"]["errorbench"].get("zeroshot_models", []):
        add("ragcritic", 2, "errorbench", f"zeroshot-{m}")
    names = {"ragcritic": rc_names(cfg), "dynakrag": [("train_controller", 1)] + dk_names(cfg)}
    for paper in PAPERS:
        for ds in paper_datasets(cfg, paper):
            if datasets and ds not in datasets:
                continue
            for name, tier in names[paper]:
                add(paper, tier, ds, name)
    return sorted(out, key=lambda j: (j["tier"], j["paper"] != "ragcritic"))


def estimate_job(cfg: dict, paper: str, dataset: str, name: str) -> float:
    """Chamadas ao LLM de um job (estimativa grosseira)."""
    dk, rc = cfg["dynakrag"], cfg["ragcritic"]
    if paper == "ragcritic":
        if dataset == "errorbench":
            eb = rc["errorbench"]
            nb = (eb.get("n_correct") or 950) + (eb.get("n_error") or 950)
            return nb * (2 if name == "rag-critic-3b" else 1)       # original + recuo trocado
        return rc["n_eval"] * _RC_CALLS.get(name, 3)
    if paper != "dynakrag":
        raise ValueError(f"artigo desconhecido: {paper}")
    if name == "train_controller":
        return dk["n_train"] * 11.5
    if name.startswith("fixed_k"):
        return dk["n_eval"]
    if name.startswith("dynakrag_cap"):
        return dk["n_eval"] * 2.5
    extra = dk["baselines"]["adaptive_rag"]["n_train"] * 7 if name == "adaptive_rag" else 0
    return dk["n_eval"] * _DK_CALLS.get(name, 4) + extra


def estimate_calls(cfg: dict) -> dict:
    """Estimativa GROSSEIRA de chamadas ao LLM por tier."""
    out = {"tier1": 0.0, "tier2": 0.0, "tier3": 0.0}
    for j in planned_jobs(cfg):
        out[f"tier{j['tier']}"] += estimate_job(cfg, j["paper"], j["dataset"], j["name"])
    return {k: int(v) for k, v in out.items()}


def run(cfg: dict) -> int:
    import httpx

    problems = 0
    print(f"Python {sys.version.split()[0]} · {platform.system()} {platform.release()} · perfil {cfg['profile']}")
    for mod, required in PACKAGES:
        try:
            importlib.import_module(mod)
            print(f"  [ok] {mod}")
        except ImportError:
            print(f"  [{'FALTA' if required else 'opcional'}] {mod}")
            problems += required
    if cfg["llm"]["backend"] != "ollama":
        print(f"backend {cfg['llm']['backend']}: nada mais a checar.")
        return problems
    base = cfg["llm"]["base_url"].rstrip("/")
    try:
        ver = httpx.get(f"{base}/api/version", timeout=10).json().get("version")
        tags = httpx.get(f"{base}/api/tags", timeout=10).json()
    except Exception as e:  # noqa: BLE001
        print(f"[ERRO] o Ollama não respondeu em {base} ({type(e).__name__}). Abra o app do Ollama ou rode `ollama serve`.")
        return problems + 1
    print(f"Ollama {ver} em {base}")
    have = {x["name"].lower() for x in tags.get("models", [])} | {x.get("model", "").lower() for x in tags.get("models", [])}
    latencies = []
    for role, model in _models_needed(cfg).items():
        ok = model.lower() in have or f"{model}:latest".lower() in have
        print(f"  [{'ok' if ok else 'FALTA'}] {role:28s} {model}" + ("" if ok else f"   →  ollama pull {model}"))
        if not ok:
            problems += 1
            continue
        if role not in ("generator", "critic"):
            continue
        prompt = f"{_SAMPLE_DOCS}\n\nQuestion: what does the fox do?\nAnswer in one short sentence."
        body = {"model": model, "stream": False, "keep_alive": cfg["llm"]["keep_alive"],
                "options": {"temperature": 0, "num_ctx": cfg["llm"]["num_ctx"], "num_predict": 40, "seed": 13}}
        if role == "critic":
            body.update(prompt=render_chatml([{"role": "user", "content": prompt}]), raw=True)
            url = f"{base}/api/generate"
        else:
            body["messages"] = [{"role": "user", "content": prompt}]
            url = f"{base}/api/chat"
        httpx.post(url, json=body, timeout=900)  # aquece (carrega o modelo)
        body["options"]["seed"] = 14              # evita reaproveitar a resposta idêntica
        t0 = time.time()
        r = httpx.post(url, json=body, timeout=900).json()
        dt = time.time() - t0
        latencies.append(dt)
        pe = r.get("prompt_eval_count", 0) / max(1e-9, r.get("prompt_eval_duration", 1) / 1e9)
        ge = r.get("eval_count", 0) / max(1e-9, r.get("eval_duration", 1) / 1e9)
        ps = httpx.get(f"{base}/api/ps", timeout=10).json().get("models", [])
        info = next((x for x in ps if model.lower() in (str(x.get("name")).lower(), str(x.get("model")).lower())), None)
        gpu = ""
        if info and info.get("size"):
            frac = info.get("size_vram", 0) / info["size"]
            gpu = f" · {frac:.0%} na GPU ({info.get('size_vram', 0) / 2 ** 30:.1f}/{info['size'] / 2 ** 30:.1f} GiB)"
            if frac < 0.95:
                print("        ⚠ o modelo não coube inteiro na GPU: tudo fica bem mais lento. Feche outros programas que usam a GPU.")
        print(f"        chamada típica (~1,2k tokens de prompt): {dt:.1f}s · prompt {pe:.0f} tok/s · geração {ge:.1f} tok/s{gpu}")
    # embeddings para indexar o corpus (uma vez por dataset): GPU pelo Ollama quando o modelo existe lá
    from .config import PAPERS, retrieval_spec
    from .core.retrieval import OLLAMA_EMBED_MODELS, _cuda_available

    dense = {retrieval_spec(cfg, p)["model"] for p in PAPERS if retrieval_spec(cfg, p).get("kind") == "dense"}
    backend = str(cfg["retrieval"].get("embed_backend") or "auto")
    for model in sorted(dense):
        tag = OLLAMA_EMBED_MODELS.get(model.lower())
        if _cuda_available() and backend in ("auto", "st"):
            print(f"  [ok] índice {model}: GPU pelo torch (CUDA)")
        elif tag and backend in ("auto", "ollama"):
            ok = tag in have or f"{tag}:latest" in have
            print(f"  [{'ok' if ok else 'opcional'}] índice {model}: " + (f"GPU pelo Ollama ({tag})" if ok else
                  f"na CPU, devagar. Para usar a GPU:  ollama pull {tag}   (o prepare também tenta baixar sozinho)"))
        else:
            print(f"  [ok] índice {model}: na CPU (este modelo não existe no Ollama; é feito uma vez só por dataset)")
    # logprobs (FLARE)
    try:
        r = httpx.post(f"{base}/api/generate", timeout=300, json={
            "model": cfg["models"]["generator"], "prompt": "Say hi.", "stream": False, "logprobs": True,
            "options": {"num_predict": 4, "temperature": 0, "num_ctx": cfg["llm"]["num_ctx"]}}).json()
        print(f"  logprobs (FLARE): {'suportado' if r.get('logprobs') else 'NÃO suportado — FLARE usa o modo sempre-busca; atualize o Ollama'}")
    except Exception:  # noqa: BLE001
        print("  logprobs (FLARE): não foi possível testar")
    calls = estimate_calls(cfg)
    if latencies:
        per = sum(latencies) / len(latencies)
        tot = sum(calls.values())
        acc, parts = 0.0, []
        for t in ("tier1", "tier2", "tier3"):
            acc += calls[t] * per / 3600
            parts.append(f"até o {t.replace('tier', 'tier ')}: ~{acc:.1f} h")
        print(f"Estimativa grosseira do perfil '{cfg['profile']}': ~{tot:,} chamadas × {per:.1f}s ≈ {tot * per / 3600:.1f} h "
              f"({' · '.join(parts)}). O cache torna as repetições gratuitas.")
    print("OK" if problems == 0 else f"{problems} problema(s) — resolva antes de rodar o benchmark.")
    return problems
