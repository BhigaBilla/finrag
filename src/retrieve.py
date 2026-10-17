"""BM25, dense, hybrid (reciprocal rank fusion) and reranked retrieval.

Build the indexes once with `python -m src.retrieve --build`; afterwards
`Retriever()` loads them from data/index/.
"""
import argparse
import json
import pickle
import re

import numpy as np

from .config import CHUNKS, INDEX

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANKERS = {
    "bge": "BAAI/bge-reranker-base",                    # 278M params, ~37 s/query on laptop CPU (pool 30)
    "minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2",   # 22M params
}
# bge v1.5 retrieval convention: instruction prefix on queries, not passages.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RRF_K = 60

STOPWORDS = set("""a an and are as at be by for from has have in is it its of on or
that the this to was were will with which our we during year""".split())


def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if t not in STOPWORDS]


def rrf(rankings, k=RRF_K):
    """Fuse ranked lists of ids: score(id) = sum over lists of 1 / (k + rank)."""
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def load_chunks():
    with open(CHUNKS, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build():
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    INDEX.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    texts = [c["text"] for c in chunks]
    print(f"{len(chunks)} chunks")

    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with open(INDEX / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    print("BM25 built")

    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True, convert_to_numpy=True)
    np.save(INDEX / "dense.npy", emb.astype(np.float32))
    print(f"dense embeddings {emb.shape}")


class Retriever:
    def __init__(self, rerank="minilm", rerank_pool=20, rerank_max_length=256):
        from sentence_transformers import SentenceTransformer

        self.chunks = load_chunks()
        self.ids = [c["id"] for c in self.chunks]
        self.by_id = {c["id"]: c for c in self.chunks}
        self.company = np.array([c["company"] for c in self.chunks])
        self.fy = np.array([c["fy"] for c in self.chunks])
        with open(INDEX / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        self.emb = np.load(INDEX / "dense.npy")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = None
        self.rerank_pool = rerank_pool
        if rerank:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANKERS[rerank], max_length=rerank_max_length)

    def _mask(self, company, fy):
        m = np.ones(len(self.chunks), dtype=bool)
        if company:
            m &= self.company == company
        if fy:
            m &= self.fy == fy
        return m

    def _top(self, scores, mask, k):
        scores = np.where(mask, scores, -np.inf)
        idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [self.ids[i] for i in idx if np.isfinite(scores[i])]

    def bm25_search(self, query, k=10, company=None, fy=None):
        return self._top(self.bm25.get_scores(tokenize(query)), self._mask(company, fy), k)

    def dense_search(self, query, k=10, company=None, fy=None):
        q = self.embedder.encode(QUERY_PREFIX + query, normalize_embeddings=True)
        return self._top(self.emb @ q, self._mask(company, fy), k)

    def hybrid_search(self, query, k=10, company=None, fy=None, pool=50):
        fused = rrf([self.bm25_search(query, pool, company, fy),
                     self.dense_search(query, pool, company, fy)])
        return fused[:k]

    def rerank_search(self, query, k=10, company=None, fy=None):
        cands = self.hybrid_search(query, self.rerank_pool, company, fy)
        scores = self.reranker.predict([(query, self.by_id[c]["text"]) for c in cands])
        order = np.argsort(-np.asarray(scores))
        return [cands[i] for i in order[:k]]

    def search(self, query, method="rerank", **kw):
        return getattr(self, f"{method}_search")(query, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--query")
    ap.add_argument("--company")
    ap.add_argument("--fy")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--reranker", default="minilm", choices=list(RERANKERS))
    args = ap.parse_args()
    if args.build:
        build()
    if args.query:
        r = Retriever(rerank=args.reranker)
        for method in ["bm25", "dense", "hybrid", "rerank"]:
            print(f"\n== {method}")
            for cid in r.search(args.query, method, k=args.k, company=args.company, fy=args.fy):
                print(f"  {cid}: {r.by_id[cid]['text'][:160]}")


if __name__ == "__main__":
    main()
