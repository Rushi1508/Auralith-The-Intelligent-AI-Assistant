### Sweep: rerank

_4 configurations. Retrieval-only (no generator, no judge)._

| config | chunks | size/ovl | k | extras | recall@k | hit@k | MRR | prec@k |
|---|---|---|---|---|---|---|---|---|
| prod-norerank | 132 | 500/100 | 6 | — | **0.8846** | 0.9231 | 0.6917 | 0.2308 |
| prod-rr10 | 132 | 500/100 | 6 | rerank←10 | 0.8462 | 0.8462 | 0.6551 | 0.2564 |
| prod-rr20 | 132 | 500/100 | 6 | rerank←20 | 0.8654 | 0.8846 | **0.7051** | 0.2692 |
| prod-rr30 | 132 | 500/100 | 6 | rerank←30 | 0.8654 | 0.8846 | **0.7051** | 0.2628 |
