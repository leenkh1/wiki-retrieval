"""Preprocessing and chunking for the dense index.

Three chunking modes (lecture: fixed-size vs. content-aware), selectable via
CHUNK_MODE so they can be compared empirically:

  * "fixed"     - sliding word windows of CHUNK_WORDS with CHUNK_OVERLAP.
  * "paragraph" - split on blank-line / newline boundaries (content-aware);
                  over-long paragraphs fall back to fixed windows.
  * "sentence"  - split each paragraph into sentences (finest granularity).

For this corpus the pages are lists of short, self-contained factual
sentences, so a content-aware split embeds each fact on its own vector instead
of blending ~10 facts into one 200-word window. The title is prepended to every
chunk so each retrieval unit keeps its topical anchor.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

CHUNK_MODE = "paragraph"   # "fixed" | "paragraph" | "sentence"
CHUNK_WORDS = 200          # max words per chunk (cap in every mode)
CHUNK_OVERLAP = 40         # used by the fixed-size mode / long-unit fallback

_PARA_SPLIT = re.compile(r"\n\s*\n+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def _fixed_windows(words: List[str], size: int, overlap: int) -> List[str]:
    if len(words) <= size:
        return [" ".join(words)] if words else []
    step = max(1, size - overlap)
    pieces: List[str] = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return pieces


def _cap(unit: str) -> List[str]:
    """Keep a unit whole unless it exceeds the word cap, then window it."""
    words = unit.split()
    if len(words) <= CHUNK_WORDS:
        return [unit]
    return _fixed_windows(words, CHUNK_WORDS, CHUNK_OVERLAP)


def _split_content(content: str, mode: str) -> List[str]:
    if mode == "fixed":
        return _fixed_windows(content.split(), CHUNK_WORDS, CHUNK_OVERLAP)

    paragraphs = [p.strip() for p in _PARA_SPLIT.split(content) if p.strip()]
    if mode == "sentence":
        units: List[str] = []
        for para in paragraphs:
            units.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    else:  # "paragraph"
        units = paragraphs

    pieces: List[str] = []
    for unit in units:
        pieces.extend(_cap(unit))
    return pieces


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """Split one corpus entry into title-anchored retrieval units."""
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()

    pieces = _split_content(content, CHUNK_MODE) or ([title] if title else [])
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