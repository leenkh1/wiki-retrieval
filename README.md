# Section B — Wikipedia Retrieval Pipeline

This project implements an end-to-end retrieval pipeline over a Wikipedia-style corpus.

The required API is:

```python
def run(queries: list[str]) -> list[list[int]]:
```

For each query, the system returns a ranked list of relevant `page_id` values.

The pipeline uses:

* Dense retrieval with `sentence-transformers/all-MiniLM-L6-v2`
* Lexical retrieval with BM25
* Score fusion between dense and lexical scores
* Prebuilt artifacts loaded from `artifacts/`

The grader should not rebuild the index. `run()` loads the submitted artifacts directly.

## Pipeline

| Stage       | File          | Description                                                                                                              |
| ----------- | ------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Chunking    | `chunk.py`    | Splits each page into overlapping text chunks and prepends the page title to each chunk.                                 |
| Embedding   | `embed.py`    | Encodes chunks and queries with `sentence-transformers/all-MiniLM-L6-v2`. Embeddings are L2-normalized.                  |
| Index build | `index.py`    | Builds and saves dense chunk vectors and a page-level BM25 lexical index.                                                |
| Retrieval   | `retrieve.py` | Searches dense vectors with FAISS, max-pools chunk scores to page scores, computes BM25 scores, then fuses both signals. |
| API         | `main.py`     | Exposes the required `run(queries)` function used by the autograder.                                                     |

## Setup

```bash
pip install -r requirements.txt
```

The corpus is stored in:

```text
data/Wikipedia Entries/
```

Public queries are stored in:

```text
data/public_queries.json
```

## Build the index offline

The index is built offline and saved under `artifacts/`.

```bash
python scripts/build_index.py
```

This step is not run by the grader. It is only for local development or rebuilding artifacts.

## Submitted artifacts

The following files are required in the GitHub repository:

| Artifact                      | Format                   | Description                                                     |
| ----------------------------- | ------------------------ | --------------------------------------------------------------- |
| `artifacts/index_vectors.npy` | NumPy array              | Dense MiniLM chunk embeddings.                                  |
| `artifacts/index_meta.json`   | JSON                     | Metadata mapping each vector row to a `page_id` and `chunk_id`. |
| `artifacts/lexical_index.npz` | NumPy compressed archive | BM25 lexical index arrays.                                      |
| `artifacts/lexical_meta.json` | JSON                     | BM25 vocabulary, page IDs, and scoring parameters.              |

These files are loaded by `run()` at query time. The grader does not call `scripts/build_index.py`.

Large artifact files are tracked with Git LFS.

## Public self-test

Run:

```bash
python scripts/eval_public.py
```

This evaluates the system on the public queries and prints mean NDCG@10.

Example result from our current submitted artifacts:

```text
public_queries=29
mean_ndcg@10=0.3342
query_phase_time≈2.9s
```

## Retrieval design

The system combines two complementary signals:

1. Dense MiniLM retrieval finds semantically similar chunks.
2. BM25 lexical retrieval helps with exact words, names, dates, and numbers.

Dense chunk scores are reduced to page-level scores using max pooling. Dense and BM25 scores are normalized per query and combined using a weighted score.

Important tunable values:

* `retrieve.ALPHA`: balance between dense and BM25 scores.
* `retrieve.DENSE_CHUNK_POOL`: number of dense chunks retrieved before page aggregation.
* `chunk.CHUNK_WORDS`: maximum chunk size.
* `chunk.CHUNK_OVERLAP`: overlap between neighboring chunks.
* `index.BM25_K1` and `index.BM25_B`: BM25 scoring parameters.

Changing chunking or rebuilding artifacts requires rerunning:

```bash
python scripts/build_index.py
```

## Video

Presentation link: *add link here before final submission*