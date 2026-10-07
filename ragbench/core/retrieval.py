"""Retrieval sobre o corpus de cada dataset: BM25 (CPU, sem download) e denso (BGE / E5 na CPU).

* DynaKRAG usa BGE-large-en-v1.5 com índice FAISS *flat* (produto interno exato). Aqui o índice
  flat é uma matriz numpy normalizada: para dezenas de milhares de parágrafos, ``E @ q`` é tão
  exato quanto o FAISS flat e dispensa a dependência.
* RAG-Critic usa E5-base-v2 (via FlashRAG) com os prefixos ``query:`` / ``passage:``.
* As CONSULTAS são sempre codificadas na CPU: os 4 GB da GPU ficam para o LLM no Ollama.
* A INDEXAÇÃO (milhares de parágrafos de uma vez) usa a GPU quando dá — ``retrieval.embed_backend``:
    - ``auto`` (padrão): torch com CUDA, se houver; senão o mesmo modelo servido pelo Ollama
      (hoje só o BGE-large tem equivalente oficial, ``bge-large``), conferido contra a versão da CPU
      numa amostra; senão CPU.
    - ``st``: sentence-transformers (CPU, ou CUDA se o torch tiver);  ``ollama``: força o Ollama.
  Antes de usar a GPU, os LLMs carregados no Ollama são descarregados para liberar memória.
* ``hash`` é um embedding falso (feature hashing) usado apenas nos testes.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import re
import threading
import time
from collections import OrderedDict

import numpy as np

from .io import ensure_dir, get_logger, read_json, write_json

_TOKEN = re.compile(r"(?u)\b\w+\b")


def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()


# ============================================================================ BM25
class BM25Index:
    """BM25 vetorizado com matriz esparsa (k1/b nos padrões do Pyserini)."""

    kind = "bm25"

    def __init__(self, k1: float = 0.9, b: float = 0.4):
        self.k1, self.b = k1, b
        self.vec = None
        self.W = None
        self.idf = None

    def build(self, docs: list[dict]) -> "BM25Index":
        from scipy.sparse import csr_matrix
        from sklearn.feature_extraction.text import CountVectorizer

        self.vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", stop_words="english")
        tf = self.vec.fit_transform([f"{d['title']} {d['text']}" for d in docs]).tocsc().astype(np.float32)
        n = tf.shape[0]
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = self.k1 * (1 - self.b + self.b * dl / max(1e-9, dl.mean()))
        tf = tf.tocoo()
        data = tf.data * (self.k1 + 1) / (tf.data + norm[tf.row])
        self.W = csr_matrix((data, (tf.row, tf.col)), shape=tf.shape).tocsc()
        return self

    def scores(self, query: str) -> np.ndarray:
        q = self.vec.transform([query])
        cols = q.indices
        if len(cols) == 0:
            return np.zeros(self.W.shape[0], dtype=np.float32)
        return np.asarray(self.W[:, cols] @ self.idf[cols]).ravel()

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "BM25Index":
        with open(path, "rb") as f:
            return pickle.load(f)


# ============================================================================ encoders densos
def query_prefix(model_name: str) -> tuple[str, str]:
    m = model_name.lower()
    if "e5" in m:
        return "query: ", "passage: "
    if "bge" in m and "m3" not in m:
        return "Represent this sentence for searching relevant passages: ", ""
    return "", ""


class HashEncoder:
    """Embedding falso e determinístico (bag-of-words com hashing) — só para testes."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 64, **_) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _TOKEN.findall(t.lower()):
                if w in ("query", "passage", "represent", "sentence", "searching", "relevant", "passages"):
                    continue
                hsh = int(hashlib.md5(w.encode()).hexdigest()[:8], 16)
                out[i, hsh % self.dim] += 1.0 if (hsh >> 8) % 2 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


# modelos de embedding que existem na biblioteca oficial do Ollama com os MESMOS pesos
OLLAMA_EMBED_MODELS = {"baai/bge-large-en-v1.5": "bge-large", "baai/bge-m3": "bge-m3", "hash": "hash"}


class OllamaEncoder:
    """Embeddings pelo Ollama (``/api/embed``), que usa a GPU. Só para indexar em lote."""

    def __init__(self, base_url: str, model: str, timeout: float = 900.0):
        self.base_url, self.model, self.timeout = base_url.rstrip("/"), model, timeout

    def encode(self, texts: list[str], batch_size: int = 32, **_) -> np.ndarray:
        import httpx

        out = []
        with httpx.Client(timeout=self.timeout) as cli:
            for i in range(0, len(texts), batch_size):
                batch = list(texts[i:i + batch_size])     # já cortados por ``DenseIndex._truncate_like_st``
                r = cli.post(f"{self.base_url}/api/embed", json={"model": self.model, "input": batch, "truncate": True,
                                                                  "keep_alive": "2m"})
                r.raise_for_status()
                emb = r.json().get("embeddings")
                if not emb or len(emb) != len(batch):
                    raise RuntimeError(f"resposta inesperada do Ollama em /api/embed ({self.model})")
                out.extend(emb)
        arr = np.asarray(out, dtype=np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)


def ollama_free_vram(base_url: str, keep: str | None = None) -> list[str]:
    """Descarrega os modelos que o Ollama mantém na GPU (menos ``keep``). Não falha se o Ollama não responder."""
    import httpx

    freed = []
    try:
        with httpx.Client(timeout=30) as cli:
            for m in cli.get(f"{base_url.rstrip('/')}/api/ps").json().get("models", []):
                name = m.get("name") or m.get("model")
                if not name or (keep and name.split(":")[0] == keep.split(":")[0]):
                    continue
                cli.post(f"{base_url.rstrip('/')}/api/generate", json={"model": name, "keep_alive": 0})
                freed.append(name)
    except Exception:  # noqa: BLE001
        pass
    return freed


def _ollama_has(base_url: str, model: str, pull: bool = True) -> bool:
    import httpx

    try:
        with httpx.Client(timeout=20) as cli:
            tags = cli.get(f"{base_url.rstrip('/')}/api/tags").json().get("models", [])
            names = {str(t.get("name", "")).lower() for t in tags} | {str(t.get("model", "")).lower() for t in tags}
            if model.lower() in names or f"{model}:latest".lower() in names:
                return True
        if not pull:
            return False
        get_logger().info(f"baixando o modelo de embedding '{model}' no Ollama (uma vez só, ~670 MB)...")
        with httpx.Client(timeout=3600) as cli:
            # "model" nas versões atuais do Ollama, "name" nas antigas
            r = cli.post(f"{base_url.rstrip('/')}/api/pull", json={"model": model, "name": model, "stream": False})
            return r.status_code == 200 and "error" not in r.text.lower()[:200]
    except Exception:  # noqa: BLE001
        return False


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


_ENCODERS: dict = {}
_ENC_LOCK = threading.Lock()


def get_encoder(model_name: str, device: str = "cpu", max_seq_length: int = 256):
    with _ENC_LOCK:
        if model_name in _ENCODERS:
            return _ENCODERS[model_name]
        if model_name == "hash":
            enc = HashEncoder()
        else:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:  # pragma: no cover
                raise RuntimeError("retrieval denso precisa de `pip install sentence-transformers` "
                                   "(ou use --set retrieval.kind=bm25)") from e
            enc = SentenceTransformer(model_name, device=device)
            enc.max_seq_length = max_seq_length
        _ENCODERS[model_name] = enc
        return enc


class DenseIndex:
    kind = "dense"

    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 32, max_seq_length: int = 256):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.E: np.ndarray | None = None
        self.qp, self.pp = query_prefix(model_name)
        self._qcache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def _enc(self):
        return get_encoder(self.model_name, self.device, self.max_seq_length)

    def build(self, docs: list[dict], backend: str = "st", ollama_url: str = "http://localhost:11434",
              label: str = "") -> "DenseIndex":
        """Codifica o corpus. ``backend``: st | ollama | auto (ver o cabeçalho do módulo)."""
        log = get_logger()
        texts = [self.pp + f"{d['title']}\n{d['text']}" for d in docs]
        self.backend_used = "st-cpu"
        emb = None
        if self.model_name != "hash" or backend == "ollama":
            if backend in ("auto", "st") and self.model_name != "hash" and _cuda_available():
                emb = self._bulk_cuda(texts, ollama_url, label)
            if emb is None and backend in ("auto", "ollama"):
                emb = self._bulk_ollama(texts, ollama_url, label, check=(backend == "auto"))
        if emb is None:
            if self.model_name != "hash":
                log.info(f"{label}embeddings {self.model_name} de {len(texts)} parágrafos na CPU "
                         f"(feito uma vez só; pode levar vários minutos)...")
            emb = self._chunked(self._enc(), texts, self.batch_size, label, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
        self.E = np.asarray(emb, dtype=np.float32)
        return self

    @staticmethod
    def _chunked(enc, texts: list[str], batch_size: int, label: str, chunk: int = 512, **kw) -> np.ndarray:
        """Codifica em blocos e registra o progresso no log (o painel mostra essas linhas)."""
        log = get_logger()
        out, t0 = [], time.time()
        for i in range(0, len(texts), chunk):
            out.append(np.asarray(enc.encode(texts[i:i + chunk], batch_size=batch_size, **kw), dtype=np.float32))
            done = min(len(texts), i + chunk)
            if len(texts) > chunk:
                rate = done / max(1e-9, time.time() - t0)
                eta = (len(texts) - done) / max(1e-9, rate)
                log.info(f"{label}embeddings {done}/{len(texts)} ({100 * done / len(texts):.0f}%) · "
                         f"{rate:.0f} parágrafos/s · faltam ~{eta / 60:.1f} min")
        return np.concatenate(out, axis=0) if out else np.zeros((0, 1), dtype=np.float32)

    def _bulk_cuda(self, texts: list[str], ollama_url: str, label: str):
        log = get_logger()
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            freed = ollama_free_vram(ollama_url)
            log.info(f"{label}embeddings {self.model_name} de {len(texts)} parágrafos na GPU (torch CUDA)"
                     + (f"; modelos descarregados do Ollama: {', '.join(freed)}" if freed else "") + "...")
            enc = SentenceTransformer(self.model_name, device="cuda")
            enc.max_seq_length = self.max_seq_length
            try:
                enc.half()
            except Exception:  # noqa: BLE001
                pass
            emb = self._chunked(enc, texts, max(self.batch_size, 64), label, chunk=2048, normalize_embeddings=True,
                                show_progress_bar=False, convert_to_numpy=True)
            del enc
            torch.cuda.empty_cache()
            self.backend_used = "st-cuda"
            return emb
        except Exception as e:  # noqa: BLE001 — sem memória, driver, etc.: cai para o próximo backend
            log.warning(f"{label}não deu para usar a GPU pelo torch ({type(e).__name__}: {str(e)[:120]})")
            return None

    def _truncate_like_st(self, texts: list[str]) -> list[str]:
        """Corta cada texto no ponto em que o sentence-transformers cortaria (``max_seq_length`` tokens),
        para o Ollama ver exatamente o mesmo trecho. Sem tokenizer rápido, corta por palavras (aproximação)."""
        tok = getattr(self._enc(), "tokenizer", None)
        if tok is None or not getattr(tok, "is_fast", False):
            return [" ".join(t.split()[:200]) for t in texts]
        out = []
        for i in range(0, len(texts), 256):
            part = texts[i:i + 256]
            enc = tok(part, truncation=True, max_length=self.max_seq_length, return_offsets_mapping=True)
            for t, offs in zip(part, enc["offset_mapping"]):
                end = max((e for b, e in offs if e > b), default=len(t))
                out.append(t[:end])
        return out

    def _bulk_ollama(self, texts: list[str], ollama_url: str, label: str, check: bool = True):
        log = get_logger()
        tag = OLLAMA_EMBED_MODELS.get(self.model_name.lower())
        if not tag:
            return None
        try:
            if self.model_name != "hash" and not _ollama_has(ollama_url, tag):
                return None
            oenc = OllamaEncoder(ollama_url, tag)
            texts = self._truncate_like_st(texts)
            if check:
                # o índice vem do Ollama e as consultas da CPU: confere, numa amostra (os primeiros e os mais
                # longos parágrafos), que os dois dão o mesmo vetor
                order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))[:4] + list(range(min(4, len(texts))))
                sample = [texts[i] for i in dict.fromkeys(order)]
                a = np.asarray(self._enc().encode(sample, normalize_embeddings=True, convert_to_numpy=True), dtype=np.float32)
                b = oenc.encode(sample, batch_size=len(sample))
                cos = (a * b).sum(axis=1) if a.shape == b.shape else np.zeros(1)
                if a.shape != b.shape or float(cos.mean()) < 0.97 or float(cos.min()) < 0.90:
                    log.warning(f"{label}o embedding do Ollama ({tag}) não coincide com o da CPU "
                                f"(cosseno médio {float(cos.mean()):.3f}, mínimo {float(cos.min()):.3f}); usando a CPU")
                    return None
                log.info(f"{label}embedding do Ollama ({tag}) conferido contra a CPU: cosseno médio "
                         f"{float(cos.mean()):.4f} em {len(sample)} parágrafos")
            freed = ollama_free_vram(ollama_url, keep=tag)
            log.info(f"{label}embeddings {self.model_name} de {len(texts)} parágrafos na GPU (Ollama: {tag})"
                     + (f"; modelos descarregados: {', '.join(freed)}" if freed else "") + "...")
            emb = self._chunked(oenc, texts, 32, label, chunk=512)
            ollama_free_vram(ollama_url)        # devolve a GPU para o LLM
            self.backend_used = "ollama"
            return emb
        except Exception as e:  # noqa: BLE001
            log.warning(f"{label}embedding pelo Ollama falhou ({type(e).__name__}: {str(e)[:120]}); usando a CPU")
            return None

    def qvec(self, query: str) -> np.ndarray:
        with self._lock:
            if query in self._qcache:
                self._qcache.move_to_end(query)
                return self._qcache[query]
        v = self._enc().encode([self.qp + query], normalize_embeddings=True, convert_to_numpy=True)[0]
        v = np.asarray(v, dtype=np.float32)
        with self._lock:
            self._qcache[query] = v
            if len(self._qcache) > 4096:
                self._qcache.popitem(last=False)
        return v

    def scores(self, query: str) -> np.ndarray:
        return self.E @ self.qvec(query)

    def save(self, path: str) -> None:
        np.save(path, self.E.astype(np.float16))
        write_json(path + ".meta.json", {"model": self.model_name, "n": int(self.E.shape[0]), "dim": int(self.E.shape[1]),
                                         "backend": getattr(self, "backend_used", "st-cpu")})

    @staticmethod
    def load(path: str, model_name: str, device: str = "cpu", batch_size: int = 32, max_seq_length: int = 256) -> "DenseIndex":
        idx = DenseIndex(model_name, device, batch_size, max_seq_length)
        idx.E = np.load(path).astype(np.float32)
        return idx


# ============================================================================ retriever
class Retriever:
    """Busca sobre um corpus: devolve dicionários {id, title, text, score, rank}."""

    def __init__(self, docs: list[dict], index, name: str = ""):
        self.docs = docs
        self.index = index
        self.name = name or getattr(index, "kind", "?")
        self.pos = {d["id"]: i for i, d in enumerate(docs)}
        self._cache: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self.docs)

    def ranked(self, query: str, depth: int = 50) -> list[tuple[int, float]]:
        key = (query, depth)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        s = self.index.scores(query or "")
        depth = min(depth, len(s))
        if depth <= 0:
            return []
        top = np.argpartition(-s, depth - 1)[:depth]
        top = top[np.lexsort((top, -s[top]))]  # desempate estável pelo índice
        res = [(int(i), float(s[i])) for i in top]
        with self._lock:
            self._cache[key] = res
            if len(self._cache) > 20000:
                self._cache.popitem(last=False)
        return res

    def search(self, query: str, k: int, exclude_ids: set | None = None, depth: int | None = None) -> list[dict]:
        exclude_ids = exclude_ids or set()
        depth = depth or max(50, k + len(exclude_ids) + 10)
        out = []
        for rank, (i, sc) in enumerate(self.ranked(query, depth)):
            d = self.docs[i]
            if d["id"] in exclude_ids:
                continue
            out.append({**d, "score": sc, "rank": rank})
            if len(out) >= k:
                break
        return out

    def get(self, doc_id: str) -> dict:
        return self.docs[self.pos[doc_id]]


# ============================================================================ construção / carga
def index_paths(data_dir: str, dataset: str, spec: dict) -> str:
    idx_dir = ensure_dir(os.path.join(data_dir, dataset, "index"))
    if spec["kind"] == "bm25":
        return os.path.join(idx_dir, "bm25.pkl")
    return os.path.join(idx_dir, f"dense-{slug(spec['model'])}.npy")


def build_index(docs: list[dict], data_dir: str, dataset: str, spec: dict, force: bool = False) -> str:
    log = get_logger()
    path = index_paths(data_dir, dataset, spec)
    if spec["kind"] == "bm25":
        if os.path.exists(path) and not force:
            return path
        BM25Index().build(docs).save(path)
        log.info(f"[{dataset}] índice BM25 com {len(docs)} parágrafos → {path}")
        return path
    meta = read_json(path + ".meta.json", {})
    if os.path.exists(path) and not force and meta.get("n") == len(docs) and meta.get("model") == spec["model"]:
        return path
    t0 = time.time()
    idx = DenseIndex(spec["model"], "cpu", spec.get("batch_size", 32), spec.get("max_seq_length", 256))
    idx.build(docs, backend=str(spec.get("embed_backend") or "auto"),
              ollama_url=str(spec.get("ollama_url") or "http://localhost:11434"), label=f"[{dataset}] ")
    idx.save(path)
    log.info(f"[{dataset}] índice denso salvo ({idx.backend_used}, {(time.time() - t0) / 60:.1f} min) → {path}")
    return path


_RETRIEVERS: dict = {}


def load_retriever(docs: list[dict], data_dir: str, dataset: str, spec: dict) -> Retriever:
    key = (data_dir, dataset, spec["kind"], spec.get("model"))
    if key in _RETRIEVERS:
        return _RETRIEVERS[key]
    path = index_paths(data_dir, dataset, spec)
    if not os.path.exists(path):
        build_index(docs, data_dir, dataset, spec)
    if spec["kind"] == "bm25":
        idx = BM25Index.load(path)
        if idx.W.shape[0] != len(docs):
            idx = BM25Index().build(docs)
            idx.save(path)
    else:
        idx = DenseIndex.load(path, spec["model"], "cpu", spec.get("batch_size", 32), spec.get("max_seq_length", 256))
        if idx.E.shape[0] != len(docs):
            build_index(docs, data_dir, dataset, spec, force=True)
            idx = DenseIndex.load(path, spec["model"], "cpu")
    r = Retriever(docs, idx, name=f"{spec['kind']}:{spec.get('model', '')}")
    _RETRIEVERS[key] = r
    return r


def clear_cache() -> None:
    _RETRIEVERS.clear()
