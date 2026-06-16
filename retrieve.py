"""Query-time retrieval.

Two-stage pipeline:
  1. recall : MiniLM dense chunk retrieval + BM25 page retrieval.
  2. rerank : cross-encoder scores each (query, page_text) pair jointly.

This version restores the cross-encoder setup and uses max_length=384.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from embed import embed_queries
from index import LEXICAL_ARRAYS_NAME, LEXICAL_META_NAME, PAGE_TEXTS_NAME, load_index
from utils import ARTIFACTS_DIR, K_EVAL, tokenize

try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False


FIRST_STAGE_K = 50
DENSE_CHUNK_POOL = 4000

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_MAX_LENGTH = 384
RERANK_BATCH = 128

_CACHE: Optional[dict] = None
_CE: Optional[CrossEncoder] = None


def _get_cross_encoder() -> CrossEncoder:
    global _CE
    if _CE is None:
        _CE = CrossEncoder(RERANK_MODEL, max_length=RERANK_MAX_LENGTH)
    return _CE


class LexicalIndex:
    """Loaded BM25 index. score() returns one BM25 value per page."""

    def __init__(
        self,
        indptr,
        csc_doc,
        csc_tf,
        idf,
        doc_len,
        page_ids,
        vocab,
        avgdl,
        k1,
        b,
    ):
        self.indptr = indptr
        self.csc_doc = csc_doc
        self.csc_tf = csc_tf
        self.idf = idf
        self.page_ids = page_ids
        self.vocab = vocab
        self.k1 = k1
        self._len_norm = (
            1.0 - b + b * doc_len / max(avgdl, 1e-9)
        ).astype(np.float32)

    @classmethod
    def load(cls, artifacts_dir: Path) -> "LexicalIndex":
        arr = np.load(artifacts_dir / LEXICAL_ARRAYS_NAME)
        meta = json.loads(
            (artifacts_dir / LEXICAL_META_NAME).read_text(encoding="utf-8")
        )
        return cls(
            arr["indptr"],
            arr["csc_doc"],
            arr["csc_tf"],
            arr["idf"],
            arr["doc_len"],
            arr["page_ids"],
            meta["vocab"],
            float(meta["avgdl"]),
            float(meta["k1"]),
            float(meta["b"]),
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


def _load_artifacts(artifacts_dir: Optional[Path]) -> dict:
    global _CACHE

    if _CACHE is not None and artifacts_dir is None:
        return _CACHE

    root = artifacts_dir or ARTIFACTS_DIR

    vectors, chunk_page_ids = load_index(artifacts_dir)
    lexical = LexicalIndex.load(root)

    page_ids = lexical.page_ids
    pos = {int(pid): i for i, pid in enumerate(page_ids)}
    chunk_pos = np.array(
        [pos.get(int(p), -1) for p in chunk_page_ids],
        dtype=np.int64,
    )

    vectors = np.ascontiguousarray(vectors.astype(np.float32))

    faiss_index = None
    if _HAVE_FAISS and vectors.size:
        faiss_index = faiss.IndexFlatIP(vectors.shape[1])
        faiss_index.add(vectors)

    page_texts = json.loads((root / PAGE_TEXTS_NAME).read_text(encoding="utf-8"))

    bundle = {
        "vectors": vectors,
        "faiss_index": faiss_index,
        "chunk_pos": chunk_pos,
        "lexical": lexical,
        "page_ids": np.asarray(page_ids, dtype=np.int64),
        "n_pages": len(page_ids),
        "page_texts": page_texts,
    }

    if artifacts_dir is None:
        _CACHE = bundle

    return bundle


def _dense_chunk_scores(
    bundle: dict,
    query_vectors: np.ndarray,
    pool: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return dense chunk scores and chunk indices."""
    if bundle["faiss_index"] is not None:
        return bundle["faiss_index"].search(query_vectors, pool)

    sims = query_vectors @ bundle["vectors"].T
    idx = np.argpartition(-sims, kth=pool - 1, axis=1)[:, :pool]
    row = np.arange(sims.shape[0])[:, None]
    order = np.argsort(-sims[row, idx], axis=1)
    idx = idx[row, order]

    return sims[row, idx], idx


def _maxpool_to_pages(
    scores: np.ndarray,
    chunk_idx: np.ndarray,
    chunk_pos: np.ndarray,
    n_pages: int,
) -> np.ndarray:
    """Reduce one query's top chunk scores to per-page max score."""
    page_scores = np.zeros(n_pages, dtype=np.float32)

    positions = chunk_pos[chunk_idx]
    valid = positions >= 0

    np.maximum.at(page_scores, positions[valid], scores[valid])
    return page_scores


def _topk_positions(scores: np.ndarray, k: int) -> Set[int]:
    """Page positions of the top-k scores."""
    k = min(k, len(scores))
    if k <= 0:
        return set()

    idx = np.argpartition(-scores, kth=k - 1)[:k]
    return {int(x) for x in idx}


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Return ranked page_id lists for each query."""
    if not queries:
        return []

    bundle = _load_artifacts(artifacts_dir)

    page_ids = bundle["page_ids"]
    n_pages = bundle["n_pages"]
    lexical: LexicalIndex = bundle["lexical"]
    page_texts = bundle["page_texts"]

    query_vectors = embed_queries(queries)

    pool = min(DENSE_CHUNK_POOL, bundle["vectors"].shape[0]) if bundle["vectors"].size else 0
    d_scores, d_idx = (
        _dense_chunk_scores(bundle, query_vectors, pool)
        if pool else (None, None)
    )

    cross_encoder = _get_cross_encoder()

    ranked: List[List[int]] = []

    for i, query in enumerate(queries):
        dense = (
            _maxpool_to_pages(d_scores[i], d_idx[i], bundle["chunk_pos"], n_pages)
            if pool else np.zeros(n_pages, dtype=np.float32)
        )

        sparse = lexical.score(query)

        candidates = list(
            _topk_positions(dense, FIRST_STAGE_K)
            | _topk_positions(sparse, FIRST_STAGE_K)
        )

        if not candidates:
            ranked.append([])
            continue

        pairs = [
            (query, page_texts.get(str(int(page_ids[p])), ""))
            for p in candidates
        ]

        scores = np.asarray(
            cross_encoder.predict(
                pairs,
                batch_size=RERANK_BATCH,
                show_progress_bar=False,
            )
        )

        order = np.argsort(-scores)[:top_k]
        ranked.append([int(page_ids[candidates[j]]) for j in order])

    return ranked