"""Offline index build and load (not timed at grading).

Builds two artifacts that ``run()`` loads at query time:
  * dense  : index_vectors.npy + index_meta.json  (chunk embeddings)
  * lexical: lexical_index.npz + lexical_meta.json (page-level BM25)

The BM25 (lexical) build lives here rather than in its own module because no
new files may be added to the package. The query-time half of BM25 lives in
``retrieve.py``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, entry_text, iter_entries, tokenize

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"
LEXICAL_ARRAYS_NAME = "lexical_index.npz"
LEXICAL_META_NAME = "lexical_meta.json"

# BM25 hyper-parameters (saturation / length normalization).
BM25_K1 = 1.5
BM25_B = 0.75


# --------------------------------------------------------------------------- #
# Lexical (BM25) index build — numpy + stdlib only.
# Postings are stored term-major (CSC-style) so scoring a query touches only
# the columns of its few terms instead of the whole matrix.
# --------------------------------------------------------------------------- #
def build_lexical(page_texts: List[str], page_ids: List[int]) -> Dict:
    """Build a term-major BM25 index over page-level documents."""
    vocab: Dict[str, int] = {}
    postings_doc: List[List[int]] = []
    postings_tf: List[List[int]] = []
    n_docs = len(page_texts)
    doc_len = np.zeros(n_docs, dtype=np.float32)

    for d, text in enumerate(page_texts):
        local: Dict[int, int] = {}
        tokens = tokenize(text)
        doc_len[d] = len(tokens)
        for tok in tokens:
            tid = vocab.get(tok)
            if tid is None:
                tid = len(vocab)
                vocab[tok] = tid
                postings_doc.append([])
                postings_tf.append([])
            local[tid] = local.get(tid, 0) + 1
        for tid, tf in local.items():
            postings_doc[tid].append(d)
            postings_tf[tid].append(tf)

    vocab_size = len(vocab)
    indptr = np.zeros(vocab_size + 1, dtype=np.int64)
    for t in range(vocab_size):
        indptr[t + 1] = indptr[t] + len(postings_doc[t])
    nnz = int(indptr[-1])

    csc_doc = np.empty(nnz, dtype=np.int32)
    csc_tf = np.empty(nnz, dtype=np.float32)
    df = np.zeros(vocab_size, dtype=np.float32)
    for t in range(vocab_size):
        start, end = int(indptr[t]), int(indptr[t + 1])
        csc_doc[start:end] = postings_doc[t]
        csc_tf[start:end] = postings_tf[t]
        df[t] = end - start

    idf = np.log(1.0 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    avgdl = float(doc_len.mean()) if n_docs else 0.0
    return {
        "vocab": vocab,
        "indptr": indptr,
        "csc_doc": csc_doc,
        "csc_tf": csc_tf,
        "idf": idf,
        "doc_len": doc_len,
        "avgdl": avgdl,
        "page_ids": np.asarray(page_ids, dtype=np.int64),
    }


def save_lexical(index: Dict, out_dir: Path) -> None:
    np.savez(
        out_dir / LEXICAL_ARRAYS_NAME,
        indptr=index["indptr"],
        csc_doc=index["csc_doc"],
        csc_tf=index["csc_tf"],
        idf=index["idf"],
        doc_len=index["doc_len"],
        page_ids=index["page_ids"],
    )
    meta = {
        "vocab": index["vocab"],
        "avgdl": index["avgdl"],
        "k1": BM25_K1,
        "b": BM25_B,
        "num_docs": int(len(index["page_ids"])),
    }
    (out_dir / LEXICAL_META_NAME).write_text(json.dumps(meta), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Full corpus build (dense + lexical) and dense load.
# --------------------------------------------------------------------------- #
def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Embed the full corpus, build the BM25 index, and persist both.

    Returns (chunk_vectors, chunk_page_ids) where row i of the dense matrix
    corresponds to chunk_page_ids[i] (page ids repeat across their chunks).
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    print(f"Loaded {len(records)} pages.", flush=True)

    # --- dense (chunk-level) ------------------------------------------------
    chunks: List[Chunk] = chunk_corpus(records)
    print(f"Created {len(chunks)} chunks/vectors to embed.", flush=True)

    vectors = embed_texts([c.text for c in chunks])
    print("Finished embedding all chunks.", flush=True)

    chunk_page_ids = [c.page_id for c in chunks]

    np.save(out_dir / INDEX_VECTORS_NAME, vectors)
    meta = {
        "page_ids": chunk_page_ids,
        "chunk_ids": [c.chunk_id for c in chunks],
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(chunk_page_ids),
        "dim": int(vectors.shape[1]) if vectors.size else 384,
    }
    (out_dir / INDEX_META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # --- lexical (page-level BM25) -----------------------------------------
    page_texts = [entry_text(r) for r in records]
    page_ids = [int(r["page_id"]) for r in records]
    save_lexical(build_lexical(page_texts, page_ids), out_dir)

    return vectors, chunk_page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed dense vectors and the per-chunk page_id map."""
    root = artifacts_dir or ARTIFACTS_DIR
    vectors = np.load(root / INDEX_VECTORS_NAME)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids