"""Embedding utilities for dense retrieval.

This module loads the configured SentenceTransformer model once and reuses it
for all later embedding calls. Embeddings are L2-normalized so dot product can
be used as cosine similarity during retrieval.

The project uses sentence-transformers/all-MiniLM-L6-v2, whose embedding
dimension is 384.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from utils import EMBEDDING_MODEL_NAME

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Load the embedding model once and return the cached instance."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME, device="cuda")
    return _model


def embed_texts(texts: Sequence[str], *, batch_size: int = 64) -> np.ndarray:
    """Encode texts into L2-normalized float32 embeddings of shape (n, 384)."""
    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    model = get_model()
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_queries(queries: List[str], *, batch_size: int = 64) -> np.ndarray:
    """Encode search queries using the same embedding pipeline as documents."""
    return embed_texts(queries, batch_size=batch_size)