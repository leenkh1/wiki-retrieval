"""Query-time retrieval (timed portion includes query embedding).

Hybrid retrieval:
  1. dense  : MiniLM query embedding -> exact inner-product search over chunk
              vectors (FAISS IndexFlatIP) -> max-pool chunk scores to pages.
  2. lexical: BM25 score per page (exact-token signal).
  3. fuse   : per-query min-max normalise each signal, then weighted sum.

Exact (flat) dense search is used deliberately: at this corpus scale it costs
milliseconds and avoids the recall loss of approximate indexes.

The query-time half of BM25 (LexicalIndex) lives here; the build half lives in
index.py. They are split this way because no new module files may be added.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from embed import embed_queries
from index import LEXICAL_ARRAYS_NAME, LEXICAL_META_NAME, load_index
from utils import ARTIFACTS_DIR, K_EVAL, tokenize

try:  # FAISS is the intended path; numpy is a safe fallback.
    import faiss
    _HAVE_FAISS = True
except Exception:  # pragma: no cover
    _HAVE_FAISS = False

# Weight on the dense signal in the fusion (lexical gets 1 - ALPHA).
# Tune on the public set; ~0.5 is a balanced starting point.
ALPHA = 0.5
# How many top chunks to pull from the dense index before max-pooling to pages.
# Large enough that any page that could reach the top-10 has a chunk included.
DENSE_CHUNK_POOL = 2000

_CACHE: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Lexical (BM25) query-time index.
# --------------------------------------------------------------------------- #
class LexicalIndex:
    """Loaded BM25 index. ``score`` returns one BM25 value per page."""

    def __init__(self, indptr, csc_doc, csc_tf, idf, doc_len, page_ids,
                 vocab, avgdl, k1, b):
        self.indptr = indptr
        self.csc_doc = csc_doc
        self.csc_tf = csc_tf
        self.idf = idf
        self.page_ids = page_ids
        self.vocab = vocab
        self.k1 = k1
        # Precompute the length-normalisation term per document.
        self._len_norm = (1.0 - b + b * doc_len / max(avgdl, 1e-9)).astype(np.float32)

    @classmethod
    def load(cls, artifacts_dir: Path) -> "LexicalIndex":
        arr = np.load(artifacts_dir / LEXICAL_ARRAYS_NAME)
        meta = json.loads((artifacts_dir / LEXICAL_META_NAME).read_text(encoding="utf-8"))
        return cls(
            arr["indptr"], arr["csc_doc"], arr["csc_tf"], arr["idf"],
            arr["doc_len"], arr["page_ids"], meta["vocab"],
            float(meta["avgdl"]), float(meta["k1"]), float(meta["b"]),
        )

    def score(self, query: str) -> np.ndarray:
        """BM25 score for every page against one query string."""
        scores = np.zeros(len(self.page_ids), dtype=np.float32)
        k1 = self.k1
        for tok in set(tokenize(query)):
            tid = self.vocab.get(tok)
            if tid is None:
                continue
            start, end = int(self.indptr[tid]), int(self.indptr[tid + 1])
            docs = self.csc_doc[start:end]
            tf = self.csc_tf[start:end]
            denom = tf + k1 * self._len_norm[docs]
            scores[docs] += self.idf[tid] * (tf * (k1 + 1.0)) / denom
        return scores


# --------------------------------------------------------------------------- #
# Artifact loading (cached for repeated run() calls).
# --------------------------------------------------------------------------- #
def _load_artifacts(artifacts_dir: Optional[Path]) -> dict:
    global _CACHE
    if _CACHE is not None and artifacts_dir is None:
        return _CACHE
    root = artifacts_dir or ARTIFACTS_DIR
    vectors, chunk_page_ids = load_index(artifacts_dir)
    lexical = LexicalIndex.load(root)

    page_ids = lexical.page_ids
    pos = {int(pid): i for i, pid in enumerate(page_ids)}
    chunk_pos = np.array([pos.get(int(p), -1) for p in chunk_page_ids], dtype=np.int64)

    vectors = np.ascontiguousarray(vectors.astype(np.float32))
    faiss_index = None
    if _HAVE_FAISS and vectors.size:
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)

    bundle = {
        "vectors": vectors,
        "faiss_index": faiss_index,
        "chunk_pos": chunk_pos,
        "lexical": lexical,
        "page_ids": np.asarray(page_ids, dtype=np.int64),
        "n_pages": len(page_ids),
    }
    if artifacts_dir is None:
        _CACHE = bundle
    return bundle


# --------------------------------------------------------------------------- #
# Dense scoring helpers.
# --------------------------------------------------------------------------- #
def _dense_chunk_scores(bundle: dict, query_vectors: np.ndarray, pool: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, chunk_indices) of shape (n_queries, pool)."""
    if bundle["faiss_index"] is not None:
        return bundle["faiss_index"].search(query_vectors, pool)  # (D, I)
    sims = query_vectors @ bundle["vectors"].T
    idx = np.argpartition(-sims, kth=pool - 1, axis=1)[:, :pool]
    row = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[row, idx], axis=1)
    idx = idx[row, order]
    return sims[row, idx], idx


def _maxpool_to_pages(scores: np.ndarray, chunk_idx: np.ndarray,
                      chunk_pos: np.ndarray, n_pages: int) -> np.ndarray:
    """Reduce one query's top chunks to a per-page max score."""
    page_scores = np.zeros(n_pages, dtype=np.float32)
    positions = chunk_pos[chunk_idx]
    valid = positions >= 0
    np.maximum.at(page_scores, positions[valid], scores[valid])
    return page_scores


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
    alpha: float = ALPHA,
) -> List[List[int]]:
    """Return ranked page_id lists (best first) for each query."""
    if not queries:
        return []
    bundle = _load_artifacts(artifacts_dir)
    page_ids = bundle["page_ids"]
    n_pages = bundle["n_pages"]
    lexical: LexicalIndex = bundle["lexical"]

    query_vectors = embed_queries(queries)
    pool = min(DENSE_CHUNK_POOL, bundle["vectors"].shape[0]) if bundle["vectors"].size else 0

    if pool:
        d_scores, d_idx = _dense_chunk_scores(bundle, query_vectors, pool)
    else:
        d_scores = d_idx = None

    ranked: List[List[int]] = []
    for i, query in enumerate(queries):
        dense = (
            _maxpool_to_pages(d_scores[i], d_idx[i], bundle["chunk_pos"], n_pages)
            if pool else np.zeros(n_pages, dtype=np.float32)
        )
        sparse = lexical.score(query)
        fused = alpha * _minmax(dense) + (1.0 - alpha) * _minmax(sparse)

        k = min(top_k, n_pages)
        top = np.argpartition(-fused, kth=k - 1)[:k]
        top = top[np.argsort(-fused[top])]
        ranked.append([int(page_ids[p]) for p in top])
    return ranked