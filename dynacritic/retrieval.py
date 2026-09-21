"""Retrievers: BM25 (CPU, sem download) e denso (sentence-transformers + FAISS)."""
from __future__ import annotations

import os
import pickle

import numpy as np

from .utils import read_jsonl


class BM25Retriever:
    """BM25 vetorizado com matriz esparsa (k1/b nos padrões do Pyserini)."""

    kind = "bm25"

    def __init__(self, docs: list[dict], k1: float = 0.9, b: float = 0.4):
        from sklearn.feature_extraction.text import CountVectorizer

        self.docs = docs
        self.vec = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", stop_words="english")
        tf = self.vec.fit_transform([f"{d['title']} {d['text']}" for d in docs]).tocsc().astype(np.float32)
        n = tf.shape[0]
        df = np.diff(tf.indptr)
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * dl / dl.mean())
        tf = tf.tocoo()
        data = tf.data * (k1 + 1) / (tf.data + norm[tf.row])
        from scipy.sparse import csr_matrix

        self.W = csr_matrix((data, (tf.row, tf.col)), shape=tf.shape).tocsc()

    def search(self, query: str, k: int) -> list[dict]:
        q = self.vec.transform([query])
        cols = q.indices
        if len(cols) == 0:
            return []
        scores = np.asarray(self.W[:, cols] @ self.idf[cols]).ravel()
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [dict(self.docs[i], score=float(scores[i])) for i in top if scores[i] > 0]


class DenseRetriever:
    kind = "dense"

    def __init__(self, docs: list[dict], model_name: str, device: str = "cpu", index=None):
        from sentence_transformers import SentenceTransformer

        self.docs = docs
        self.model = SentenceTransformer(model_name, device=device)
        # modelos BGE v1.5 recomendam este prefixo em queries curtas
        self.q_prefix = "Represent this sentence for searching relevant passages: " if "bge" in model_name.lower() else ""
        self.index = index

    def build(self, batch_size: int = 64):
        import faiss

        emb = self.model.encode(
            [f"{d['title']}. {d['text']}" for d in self.docs], batch_size=batch_size,
            normalize_embeddings=True, show_progress_bar=True, convert_to_numpy=True,
        ).astype(np.float32)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def search(self, query: str, k: int) -> list[dict]:
        q = self.model.encode([self.q_prefix + query], normalize_embeddings=True, convert_to_numpy=True)
        scores, ids = self.index.search(q.astype(np.float32), k)
        return [dict(self.docs[i], score=float(s)) for s, i in zip(scores[0], ids[0]) if i >= 0]


# ------------------------------------------------------------------ build / load
def build_index(cfg: dict) -> None:
    docs = read_jsonl(cfg["paths"]["corpus"])
    rc = cfg["retriever"]
    os.makedirs(cfg["paths"]["index"], exist_ok=True)
    if rc["kind"] == "bm25":
        r = BM25Retriever(docs)
        with open(os.path.join(cfg["paths"]["index"], "bm25.pkl"), "wb") as f:
            pickle.dump(r, f)
    else:
        import faiss

        r = DenseRetriever(docs, rc["dense_model"], rc.get("device", "cpu"))
        r.build()
        faiss.write_index(r.index, os.path.join(cfg["paths"]["index"], "dense.faiss"))
    print(f"índice {rc['kind']} construído com {len(docs)} parágrafos")


def load_retriever(cfg: dict):
    rc = cfg["retriever"]
    if rc["kind"] == "bm25":
        with open(os.path.join(cfg["paths"]["index"], "bm25.pkl"), "rb") as f:
            return pickle.load(f)
    import faiss

    docs = read_jsonl(cfg["paths"]["corpus"])
    index = faiss.read_index(os.path.join(cfg["paths"]["index"], "dense.faiss"))
    return DenseRetriever(docs, rc["dense_model"], rc.get("device", "cpu"), index=index)
