### Sweep: hybrid

_6 configurations. Retrieval-only (no generator, no judge)._

| config | chunks | size/ovl | k | extras | recall@k | hit@k | MRR | prec@k |
|---|---|---|---|---|---|---|---|---|
| prod-dense-only | 132 | 500/100 | 6 | — | **0.8846** | 0.9231 | 0.6917 | 0.2308 |
| prod-hybrid0.00 | 132 | 500/100 | 6 | hybrid 0.00 | 0.7308 | 0.7308 | 0.5846 | 0.1987 |
| prod-hybrid0.30 | 132 | 500/100 | 6 | hybrid 0.30 | 0.7308 | 0.7308 | 0.5974 | 0.1987 |
| prod-hybrid0.50 | 132 | 500/100 | 6 | hybrid 0.50 | 0.8654 | 0.8846 | 0.6904 | 0.2372 |
| prod-hybrid0.70 | 132 | 500/100 | 6 | hybrid 0.70 | **0.8846** | 0.9231 | **0.6962** | 0.2308 |
| prod-hybrid1.00 | 132 | 500/100 | 6 | hybrid 1.00 | **0.8846** | 0.9231 | 0.6917 | 0.2308 |
