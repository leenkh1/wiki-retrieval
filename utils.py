"""Shared paths and helpers for Section B."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

STUDENT_ROOT = Path(__file__).resolve().parent
DATA_DIR = STUDENT_ROOT / "data"
ENTRIES_DIR = DATA_DIR / "Wikipedia Entries"
PUBLIC_QUERIES_PATH = DATA_DIR / "public_queries.json"
ARTIFACTS_DIR = STUDENT_ROOT / "artifacts"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K_EVAL = 10

# --- Lexical tokenisation (used by the BM25 index) -------------------------
# A small, generic stop list. Kept deliberately short so that we never drop
# rare discriminative tokens (names, places, numbers) that the queries hinge on.
STOPWORDS = frozenset(
    """
    a an the and or but if then else of to in on at by for with from into over
    is are was were be been being this that these those it its as i you he she
    they we who whom which what when where why how do does did done has have had
    not no nor so than too very can could should would may might will just about
    above below up down out off again further once here there all any both each
    few more most other some such only own same s t don
    """.split()
)

_NUM_COMMA = re.compile(r"(?<=\d),(?=\d)")          # 1,456,779 -> 1456779
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase, glue comma-separated numbers, split on non-alphanumerics,
    and drop stop words. Numbers and proper nouns survive intact."""
    text = _NUM_COMMA.sub("", text.lower())
    return [tok for tok in _TOKEN.findall(text) if tok not in STOPWORDS]


def normalize_page_id(value: Any) -> int:
    """Coerce page_id from JSON (int or numeric string) to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise ValueError(f"Invalid page_id: {value!r}")


def load_public_queries(path: Path | None = None) -> List[Dict[str, Any]]:
    """Load public evaluation queries and normalize their relevant page IDs."""
    path = path or PUBLIC_QUERIES_PATH
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row["relevant_page_ids"] = [
            normalize_page_id(pid) for pid in row["relevant_page_ids"]
        ]
    return rows


def iter_entries(entries_dir: Path | None = None) -> Iterator[Dict[str, Any]]:
    """Yield one record per JSON file in the corpus directory."""
    root = entries_dir or ENTRIES_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"Corpus directory not found: {root}. "
            "Expected student/data/Wikipedia Entries/ with one JSON file per page."
        )
    for path in sorted(root.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["page_id"] = normalize_page_id(data.get("page_id", path.stem))
        yield data


def entry_text(record: Dict[str, Any]) -> str:
    """Return the title plus content text used for indexing and reranking."""
    title = record.get("title", "")
    content = record.get("content", "")
    if title:
        return f"{title}\n\n{content}".strip()
    return str(content).strip()


def ensure_artifacts_dir() -> Path:
    """Create artifacts/ if needed and return its path."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR