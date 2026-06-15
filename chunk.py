"""Preprocessing and chunking for the dense index.

Long pages exceed MiniLM's 256-token limit, so the tail of the article would
never reach the encoder if we embedded whole pages. We therefore split the
content into overlapping word windows and prepend the title to every chunk so
each retrieval unit keeps its topical anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# ~200 words stays comfortably inside MiniLM's 256 word-piece window while
# keeping each chunk topically coherent; the overlap avoids splitting a fact
# across a boundary.
CHUNK_WORDS = 200
CHUNK_OVERLAP = 40


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _windows(words: List[str], size: int, overlap: int) -> List[str]:
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(1, size - overlap)
    pieces: List[str] = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return pieces


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into retrieval units (title-anchored chunks)."""
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()

    pieces = _windows(content.split(), CHUNK_WORDS, CHUNK_OVERLAP) or [title]
    chunks: List[Chunk] = []
    for i, piece in enumerate(pieces):
        text = f"{title}\n\n{piece}".strip() if title else piece
        chunks.append(Chunk(page_id=page_id, chunk_id=i, text=text))
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks