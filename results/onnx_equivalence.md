# ONNX (fastembed) vs torch (sentence-transformers)

Same models: `sentence-transformers/all-MiniLM-L6-v2` (embeddings) and `ms-marco-MiniLM-L-6-v2` (reranker, Xenova ONNX export). fastembed 0.9.1. `AURALITH_ML_BACKEND` selects the backend.

## Numerical equivalence (tests/test_onnx_equivalence.py, production chunking, 132 corpus chunks)

- Embeddings: dim 384, L2 norm 1, cosine torch vs onnx: min 0.99999996, mean 1.00000003.
- Reranker: 10 questions x 20 chunks, ranking order identical in 10/10 (top-6 identical 10/10).

## Retrieval, production_rerank, all 142 questions (`--no-generate`, Chroma, fresh index per backend)

Per-question retrieved (doc_id, page) lists, in order: torch == onnx == results/v2_rerank.json for 142/142.

| | recall@6 | hit@6 | MRR | precision@6 |
|---|---|---|---|---|
| torch | 0.904 | 0.9242 | 0.772 | 0.2715 |
| onnx | 0.904 | 0.9242 | 0.772 | 0.2715 |

Identical retrieval means identical generator inputs, so answer quality (answer_correctness 0.549 in v2_rerank.json) is unchanged.

## Memory and latency (Docker, `--cpus=0.1`, `AURALITH_ONNX_THREADS=1`, Postgres + Gemini, attention.pdf)

- Process RSS after ingest + first ask: about 494 MB; ask-only after restart: about 500 MB RSS (cgroup peak 680 MB incl. page cache). With `--memory=512m` the container was OOM-killed on the first /ask (ingest alone completed). It does not fit Render's 512 MB reliably.
- Ingest (15 pages, 29 chunks) at 0.1 CPU: 54-78 s. /ask at 0.1 CPU: 184-291 s (cold 291 s).
