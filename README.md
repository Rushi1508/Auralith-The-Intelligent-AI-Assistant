# Auralith

[![CI](https://github.com/Rushi1508/Auralith-The-Intelligent-AI-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/Rushi1508/Auralith-The-Intelligent-AI-Assistant/actions/workflows/ci.yml)

**A document Q&A assistant where every design decision is backed by a measurement.**

Ask questions about a PDF and get answers that cite their source passages. General
questions go through a LangGraph agent that picks between web search, code
generation, or conversation. It runs fully locally by default: no API key needed. A free Gemini key can optionally replace the local generator.

## Highlights

- **Retrieval recall up from 0.62 to 0.88.** Retrieval misses fell 78% (9/26 → 2/26) after systematic sweeps of chunking, k, hybrid search and reranking.
- **Wrongly refused answers down from 5/26 to 2/26** after a prompt change that lets the model give partial answers.
- **Reranking is now proven on 142 questions:** answer correctness 0.465 → 0.549 (95% CI [+0.02, +0.15]), false abstentions 14% → 6%. The earlier 28-question run could not show this.
- **The generator matters more than retrieval tweaks.** Swapping the local 3B for Gemini flash-lite on identical retrieval: 0.549 → 0.739 (CI [+0.12, +0.26]). Query decomposition added +0.04 on a held-out split, not distinguishable from noise.
- **The LLM-as-judge is checked against hand labels.** The first judge agreed with a human only 15/24 times; the rewritten one agrees 21/24 (87.5%). Re-scoring older runs with it showed that the earlier answer-quality gain was an artefact of the first judge.
- **Agent router errors cut from 9 to 2 across 84 requests** by a prompt fix for a bias towards web search, checked on a held-out set written before the fix (20/24 → 24/24).
- **Negative results are reported too.** Hybrid BM25 search was hypothesised, tested and rejected (+0.0045 MRR, which is noise).
- **Every number has a 95% confidence interval** from a paired bootstrap, and any claim whose interval includes zero is labelled as not established.

## Results

28 questions over 3 arXiv papers, with the papers pinned by SHA-256.

| metric | baseline | tuned | 95% CI of delta |
|---|---|---|---|
| recall@k | 0.615 | **0.885** | [+0.12, +0.44] |
| hit@k | 0.654 | **0.923** | |
| MRR | 0.510 | **0.692** | [+0.03, +0.33] |

What made the difference:

| decision | finding |
|---|---|
| Chunking 500 / 100 overlap | The original 7500-token "chunking" never split a single page. **Overlap mattered more than size.** |
| k = 6 | hit@k and MRR stop improving at 6, and k=8 adds 33% more context for one question. |
| Hybrid BM25: **rejected** | Dense retrieval already scored 1.00 on the keyword-heavy questions BM25 was meant to help. |
| Reranking 20 → 6: **on** | Slightly lower recall (loses one page) but better ranking. End to end it fixed 3 answers and broke none (hand 19 → 22–24/28), matching the 7B generator. Switched on as the default. On the 142-question v2 set the gain is proven (see below). |

Retrieval is no longer the bottleneck. The right page now reaches the model for 24 of
26 answerable questions, so the remaining errors come from the 3B local generator.

Answer correctness, all runs scored by the same fixed judge and also checked by hand:

| run | judge | hand |
|---|---|---|
| baseline | 14/28 | 13/28 |
| tuned retrieval | 14/28 | 15/28 |
| tuned + partial-answer prompt | 16/28 | 19/28 |
| same, 7B generator (qwen2.5:7b) | **22/28** | **22/28** |
| tuned + partial + reranking (3B) | 21/28 | **22–24/28** |

Better retrieval did not measurably improve answers: tuned fixes 8 baseline errors but
makes 6 new ones, all with the right page in context. The partial-answer prompt, the
7B generator and reranking each help, mostly on multi-part answers. Reranking with
the 3B model scores about the same as the 7B model, at much lower cost. But every
correctness interval still reaches zero on 28 questions when borderline answers are
failed, so no answer-quality gain is claimed. Details are in
[`results/comparison_ci.md`](results/comparison_ci.md).

### v2: 142 questions (supersedes the 28-question numbers above)

| run | answer correctness | 95% CI of delta |
|---|---|---|
| dense, 3B generator | 0.465 | n/a |
| + rerank 20 → 6 (default), 3B | 0.549 | [+0.02, +0.15] vs dense, **excludes 0** |
| same retrieval, Gemini flash-lite generator | **0.739** | [+0.12, +0.26] vs 3B, **excludes 0** |

Hit@6 rose 0.864 → 0.924 and MRR 0.641 → 0.772 with rerank. By type, cross-document
questions remain the weakest (0.29 with Gemini, n=7). Query decomposition
(`production_decompose`) gave +0.04 on a held-out split, interval [-0.01, +0.10], at twice
the latency, so it is off by default. The Gemini runs use the hosted model for generation
only; the judge, embeddings and reranker stay local. Full tables:
[`results/comparison_ci.md`](results/comparison_ci.md).

Agent routing (llama3.2:3b, hand-labelled requests):

| router prompt | main set | held-out set |
|---|---|---|
| original | 55/60 | 20/24 |
| revised | **58/60** | **24/24** |

The original prompt sent every misrouted request to web search, including code fixes
and a translation. The held-out gain clears zero (paired CI [+0.04, +0.33]). The fix
overcorrects slightly: 2 of 20 search questions now go to conversation.
With `gemini-3.5-flash-lite` as the router, the revised prompt routes the main set **60/60**
(6 replies needed a retry to parse; the held-out set was not re-run).

Full tables, per-question-type breakdowns and limitations are in
[`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and [`results/`](results/).

## Architecture

```mermaid
graph LR;
    PDF --> Chunk[Chunk 500/100] --> Chroma[(Chroma, cached per scheme)];
    Q[Question] --> Dense[Dense retriever k=6] --> RAG[Answer + cited passages];
    Chroma --> Dense;
```

```mermaid
graph TD;
    route -. SearchWeb .-> search;
    route -. WriteCode .-> code;
    route -. Converse .-> converse;
    route -. unparseable .-> recover;
    recover -. retry .-> route;
```

- The optional retrieval stages (BM25, multi-query, reranking) are switched on by config flags, which is what made the sweeps possible.
- `auralith/` has no Streamlit imports, so the evaluation runs exactly the same code as the app.
- The `recover → route` loop is why the agent is a graph rather than a chain: a malformed classification gets retried instead of crashing.

## Quickstart

```bash
pip install -r requirements.txt
ollama pull llama3.2:3b
cp .env.example .env
uvicorn auralith.api:app --port 8000
# second terminal:
streamlit run main.py   # on Windows, if `streamlit` isn't found: python -m streamlit run main.py
```

**Postgres (optional).** Set `DATABASE_URL=postgresql://user:pass@host:5432/db` (a pgvector-enabled Postgres)
and vectors are stored with pgvector and chat memory with the LangGraph Postgres checkpointer, keyed by the
`thread_id` the UI sends. Unset, it falls back to Chroma on disk and in-memory chat state (lost on restart).
Docker Compose starts Postgres and sets this for you.

The UI talks to the API over HTTP. `/ask` streams NDJSON (passages, then the full answer), not token-by-token. Image mode still runs locally.

Set `AURALITH_PROVIDER=openai` and `OPENAI_API_KEY` to use hosted models instead.

**No GPU or no Ollama?** Use a free Gemini key (aistudio.google.com) for the generator. Embeddings and the
reranker still run locally, so no Ollama is needed. In `.env`:

```
AURALITH_GENERATOR_PROVIDER=gemini
GEMINI_API_KEY=<your key>
```

The default model is `gemini-3.5-flash-lite`; override it with `AURALITH_GENERATOR_MODEL`. Free-tier
content may be used by Google to improve its products, so don't send confidential documents. The image
mode still uses Ollama. The router prompts were tuned on llama3.2:3b; with Gemini the main router set scores 60/60.

To reproduce the evaluation:

```bash
python eval/corpus/fetch_corpus.py
ollama pull qwen2.5:7b                  # judge model
python eval/run_eval.py --config production   # or --config baseline
```

Tests: `pytest -m "not slow and not ollama and not corpus"`. The suite covers the metrics, chunking, retrieval, the agent graph's recovery paths (driven by a scripted fake model) and the judge.

## Run with Docker

```bash
cp .env.example .env     # then set AURALITH_GENERATOR_PROVIDER=gemini and GEMINI_API_KEY
docker compose up --build
```

UI at http://localhost:8501, API docs at http://localhost:8000/docs.

For local models instead, start Ollama too and pull a model:

```bash
docker compose --profile ollama up --build
docker compose exec ollama ollama pull llama3.2:3b
```

The embedding and reranker models download once into the `hf` volume; vectors, the upload registry and chat state live in the `postgres` service (pgvector), in the `pgdata` volume, so documents survive an API restart and any worker can serve any request.

### Auth and workers

- `python -m auralith.registry create-key alice` prints a raw key once plus an `alice:<sha256>` entry. Put entries in `.env` as `AURALITH_API_KEYS=alice:<hash>,bob:<hash>`; clients send `X-API-Key: <raw key>` (UI: sidebar field or `AURALITH_API_KEY`). `/ingest`, `/ask`, `/chat`, `/documents` require it; `/health` does not.
- Documents and chat threads are per user: another user's `doc_id` is a 404, and thread ids are namespaced by user.
- Auth fails open: if `AURALITH_API_KEYS` is unset, auth is off. Check `/health` for `"auth": true`.
- Adding or revoking keys needs `docker compose up -d api` (a plain `restart` does not re-read `env_file`).
- `AURALITH_WORKERS` (default 2) sets uvicorn workers; each loads its own models. The in-flight cap and the degraded flag in `/health` are per worker, so the effective cap is workers x `AURALITH_MAX_INFLIGHT`.

### Deploy (free)

`docker-compose.prod.yml` layers production settings on the same stack: Caddy with automatic HTTPS as the only public service, Gemini as the generator (no Ollama), restart policies, and required secrets (compose refuses to start without them). [docs/deploy-oracle.md](docs/deploy-oracle.md) walks through it on an Oracle Cloud Always Free ARM VM, with a free `sslip.io` hostname, at $0.

### Logs and tracing

- Every response has an `X-Request-ID` header (sent by the client, or generated). The API logs one JSON line per request with `rid`, `method`, `path`, `status`, `ms` and `user`; unhandled errors log a traceback and return `500 {"detail": "Internal error", "request_id": ...}`. `docker compose logs -f api` to follow; `AURALITH_LOG_LEVEL` sets the level.
- LangSmith: set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` and optionally `LANGSMITH_PROJECT` in `.env`, then `docker compose up -d api`. LangChain traces every model and retriever call; no code change. This sends prompts and document text to LangSmith.

### End-to-end check

PowerShell shown; use `curl.exe`, not the `curl` alias.

1. `docker compose up -d --build`
2. `docker compose exec api python -m auralith.registry create-key alice` (and `bob`); put both `user:<hash>` entries in `.env` as `AURALITH_API_KEYS`.
3. `docker compose up -d api ui`; `curl.exe localhost:8000/health` shows `"auth": true`.
4. Ingest with alice's key, then ask with the returned `doc_id`:
   ```powershell
   curl.exe -H "X-API-Key: <alice key>" -F "file=@x.pdf;type=application/pdf" http://localhost:8000/ingest
   Invoke-RestMethod -Method Post http://localhost:8000/ask -Headers @{"X-API-Key"="<alice key>"} -ContentType application/json -Body '{"doc_id":"<id>","question":"What is this about?"}'
   ```
5. Chat as alice: `Invoke-RestMethod -Method Post http://localhost:8000/chat -Headers @{"X-API-Key"="<alice key>"} -ContentType application/json -Body '{"query":"remember pineapple","thread_id":"t1"}'`
6. `docker compose restart api`; `/ask` again with no upload: it still answers.
7. Bob's `/ask` on alice's `doc_id` returns 404; no key returns 401.
8. Bob's `/chat` on `t1` does not know pineapple; alice's does.
9. `GET /documents` lists only your docs; `DELETE /documents/<id>` removes the doc and its vectors (204; 404 for someone else's). The PDF page lets you pick or delete a stored doc without re-uploading.

`AURALITH_API_KEY` in the UI container is shown to every browser that opens the UI: use it for dev or single-user setups only; otherwise leave it unset and type the key in the sidebar.

## Limits and degraded mode

- Caps: uploads over `AURALITH_MAX_UPLOAD_MB` (25) return 413; questions/queries over `AURALITH_MAX_QUESTION_CHARS` (2000) return 422; more than `AURALITH_MAX_INFLIGHT` (4, per worker) concurrent ingest/ask/chat requests return 503 with `Retry-After`.
- If vector retrieval fails (e.g. Postgres down), `/ask` falls back to BM25 over the same chunks and flags the answer as degraded.
- `/health` always returns 200; `status` is `degraded` and `checks` lists what is wrong (database unreachable, last retrieval fell back).
- `PGCONNECT_TIMEOUT` (default 5s) bounds how long a dead database can stall a request.
- CPU-only local Ollama users should set `AURALITH_TIMEOUT=600`; the default 120s is too short for a 3B model on CPU.

## Limits

This runs locally or via Docker Compose; it is not a deployed product. The evaluation uses 142 questions on
3 NLP papers (the original 28 are superseded); small subgroups such as cross-document (n=7) are
too small to conclude from, and the judge is a local 7B model checked against only 24 hand labels. Settings were tuned
one at a time rather than over a full grid, and answer quality with the default 3B CPU
model is limited (0.55 vs 0.74 with a hosted model).

---

*Rebuilt from a 2024 prototype that didn't run: it had a broken import, a hardcoded API key, and a pinned `langchain` version that was never published. Commit 1 keeps it verbatim so you can see the before/after.*
