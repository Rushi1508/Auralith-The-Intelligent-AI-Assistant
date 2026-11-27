# Gemini embeddings vs the local MiniLM + cross-encoder pipeline

Question: can a 512 MB host drop the local models (hosted embeddings, no reranker) without losing retrieval quality? Gate: recall@6 >= 0.904 and MRR >= 0.772, the `production_rerank` baseline ([v2_rerank.json](v2_rerank.json)). 142 questions, `--no-generate`, Chroma, model `gemini-embedding-001` (768-d, RETRIEVAL_DOCUMENT / RETRIEVAL_QUERY, L2-normalised). Every run: 142 questions, 0 errors.

| config | embeddings | rerank | recall@6 | hit@6 | MRR | gate |
|---|---|---|---|---|---|---|
| production_rerank (baseline) | MiniLM | local cross-encoder | 0.904 | 0.9242 | 0.772 | - |
| production, dense only | MiniLM | none | - | - | 0.641 | - |
| **(a) production, dense only** | **Gemini** | **none** | **0.9388** | **0.9621** | **0.841** | **pass** |
| (b) production_hybrid (BM25 0.3 / dense 0.7) | Gemini | none | 0.9388 | 0.9621 | 0.8674 | pass |
| (c) production_rerank | Gemini | local cross-encoder | 0.9514 | 0.9697 | 0.8206 | pass (quality data only) |

Winner by the "cheapest passing" rule: (a), so `AURALITH_RERANK_PROVIDER=none` on Render. (b) is higher on MRR (+0.026) at the cost of BM25 over the stored chunks; it became the `PRODUCTION_CONFIG` default (below). LLM listwise rerank (d) was not built or run.

## Hybrid as the production default

`PRODUCTION_CONFIG` now has `use_bm25=True, dense_weight=0.7`. Both exact production paths, 142 questions, 0 errors, `--no-generate`, Chroma (scratch runs `results/_render_hybrid.json` and `results/_local_hybrid_rerank.json`, gitignored; the Render row equals `production_hybrid`):

| path | recall@6 | hit@6 | MRR | gate |
|---|---|---|---|---|
| Render: Gemini embeddings, rerank none, hybrid pool k=6 | 0.9388 | 0.9621 | 0.8674 | pass (>= 0.9388 / 0.841) |
| Render, hybrid but 20-deep pool through the pass-through reranker | 0.9274 | 0.947 | 0.8456 | fail on recall; so `none` skips the reranker and fetches k |
| Local: MiniLM + cross-encoder, hybrid 20 -> 6 | 0.9533 | 0.9697 | 0.8203 | pass (>= 0.904 / 0.772) |

## Cost

Gemini embedding items (free-tier quota counts one item per text): 132 chunks (5 requests of up to 32) + 142 queries + 3 smoke queries, about 277 items in total for all configs; later configs reused the on-disk cache (`AURALITH_EMBED_CACHE`).

## Spot check: "optimizer" (Adam chunk, attention.pdf page 7)

Dense Gemini ranks over the full corpus (k=20):

| phrasing | rank of the Adam chunk |
|---|---|
| What optimizer was used? | 1 |
| Which optimizer did they use to train the Transformer? | 1 |
| What optimizer was used for training? | 1 |

## Caveats

- One corpus (3-4 papers), 142 questions; the margin over the gate is large but the CI is not computed.
- Indexes are tagged by model (`__gemini-embedding-001`), so MiniLM and Gemini vectors never share a collection.
