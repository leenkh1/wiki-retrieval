# Section B — Wikipedia Retrieval Pipeline

This repository implements an end-to-end retrieval pipeline over a Wikipedia-style corpus.

The required API is:

```python
def run(queries: list[str]) -> list[list[int]]:
```

For each query, the system returns a ranked list of relevant `page_id` values. Only the top 10 returned page IDs are scored by the evaluator.

## Overview

The system uses a hybrid retrieval pipeline:

* Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`
* Lexical retrieval with a page-level BM25 index
* Candidate union from dense and BM25 retrieval
* Cross-encoder reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2`
* Prebuilt artifacts loaded from `artifacts/`

The grader should not rebuild the index. The submitted repository already includes the required artifacts, and `run()` loads them directly at query time.

## Pipeline

| Stage       | File          | Description                                                                                                                                                                                    |
| ----------- | ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chunking    | `chunk.py`    | Splits each page using paragraph-based content-aware chunking, with a 200-word cap and 40-word overlap fallback for long units. The page title is prepended to each chunk.                     |
| Embedding   | `embed.py`    | Encodes chunks and queries with `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are L2-normalized.                                                                                        |
| Index build | `index.py`    | Builds and saves dense chunk vectors, dense metadata, a page-level BM25 lexical index, and page texts for reranking.                                                                           |
| Retrieval   | `retrieve.py` | Searches dense vectors with FAISS when available, max-pools chunk scores to page-level scores, retrieves BM25 candidates, unions both candidate sets, and reranks pages using a cross-encoder. |
| API         | `main.py`     | Exposes the required `run(queries)` function used by the evaluator.                                                                                                                            |

## Repository Structure

```text
.
??? artifacts/
??? data/
??? scripts/
?   ??? build_index.py
?   ??? eval_public.py
??? chunk.py
??? embed.py
??? eval.py
??? index.py
??? main.py
??? retrieve.py
??? utils.py
??? requirements.txt
??? README.md
```

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies are:

```text
numpy
sentence-transformers
faiss-cpu
torch
```

## Data

The corpus is stored in:

```text
data/Wikipedia Entries/
```

Public queries are stored in:

```text
data/public_queries.json
```

## Submitted Artifacts

The index is built offline and saved under `artifacts/`.

The following artifact files are required in the repository:

| Artifact                      | Format                   | Description                                                           |
| ----------------------------- | ------------------------ | --------------------------------------------------------------------- |
| `artifacts/index_vectors.npy` | NumPy array              | Dense MiniLM chunk embeddings.                                        |
| `artifacts/index_meta.json`   | JSON                     | Metadata mapping each dense vector row to a `page_id` and `chunk_id`. |
| `artifacts/lexical_index.npz` | NumPy compressed archive | BM25 lexical index arrays.                                            |
| `artifacts/lexical_meta.json` | JSON                     | BM25 vocabulary, page IDs, and scoring parameters.                    |
| `artifacts/page_texts.json`   | JSON                     | Truncated page texts used by the cross-encoder reranker.              |

These files are loaded by `run()` at query time. The grader does not need to run `scripts/build_index.py`.

Large artifact files are tracked with Git LFS.

## Public Self-Test

Run:

```bash
python scripts/eval_public.py
```

This evaluates the system on the public queries and prints mean NDCG@10.

Fresh-clone test result from the current submitted artifacts:

```text
public_queries=29
mean_ndcg@10=0.4372
query_phase_time=19.68s
```

## Offline Index Build

The submitted repository already includes prebuilt artifacts. Rebuilding is only needed for local development after changing chunking, embedding, or indexing settings.

To rebuild artifacts locally:

```bash
python scripts/build_index.py
```

This creates the files under:

```text
artifacts/
```

The grading path should use the existing artifacts and should not rebuild the index.

## Retrieval Design

The retrieval pipeline uses two complementary first-stage recall methods.

First, dense retrieval embeds the query with `sentence-transformers/all-MiniLM-L6-v2` and searches the dense chunk embedding matrix. Since documents are chunked, multiple chunks can map to the same page. The system max-pools chunk scores to obtain one dense score per page.

Second, BM25 lexical retrieval scores full pages using a lightweight page-level inverted index. This helps with exact names, dates, numbers, and rare terms that may be missed by dense retrieval.

The top pages from dense retrieval and BM25 are combined into a candidate set. A cross-encoder reranker then scores each `(query, page_text)` pair jointly and returns the final ranked `page_id` list.

Important tunable values:

* `chunk.CHUNK_MODE`: selected chunking strategy.
* `chunk.CHUNK_WORDS`: maximum chunk size.
* `chunk.CHUNK_OVERLAP`: overlap used for long-unit fallback.
* `index.BM25_K1`: BM25 term-frequency saturation parameter.
* `index.BM25_B`: BM25 length-normalization parameter.
* `retrieve.FIRST_STAGE_K`: number of dense/BM25 page candidates before reranking.
* `retrieve.DENSE_CHUNK_POOL`: number of dense chunks retrieved before page aggregation.
* `retrieve.RERANK_MAX_LENGTH`: maximum input length for cross-encoder reranking.
* `retrieve.RERANK_BATCH`: reranker batch size.

## Required Entry Point

The evaluator imports and calls:

```python
from main import run

results = run(queries)
```

where `queries` is a list of strings.

The returned value is:

```python
list[list[int]]
```

Each inner list contains ranked `page_id` values, with the most relevant page first.

## Video Presentation

Presentation link: **ADD VIDEO LINK HERE BEFORE FINAL SUBMISSION**

The video explains the full pipeline: chunking, embedding, indexing, retrieval, reranking, development process, and empirical results.
