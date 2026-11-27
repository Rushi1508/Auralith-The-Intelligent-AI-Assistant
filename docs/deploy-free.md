# Deploy free, no card

Render free web service (API) + Neon (Postgres) + Streamlit Community Cloud (UI) + UptimeRobot (keep-alive) + Gemini (generator). None of them asks for a card.

Render's free instance has 512 MB RAM and 0.1 CPU, which no local model fits (the cross-encoder alone made /ask take 184 s). `Dockerfile.render` is a thin image: embeddings come from Gemini (`AURALITH_EMBEDDING_PROVIDER=gemini`, `gemini-embedding-001`), reranking is off (`AURALITH_RERANK_PROVIDER=none`), vectors live in Postgres (`DATABASE_URL` is required). Retrieval is hybrid (dense + BM25 over the stored chunks) and not worse than the old local reranked pipeline: recall@6 0.939 vs 0.904, MRR 0.867 vs 0.772 ([results/gemini_embeddings.md](../results/gemini_embeddings.md)). The local ONNX path (`AURALITH_ML_BACKEND=onnx`, [results/onnx_equivalence.md](../results/onnx_equivalence.md)) stays available for bigger hosts.

## 1. Neon (Postgres)

Create a project at neon.tech. Copy the **direct** connection string (host without `-pooler`) and keep `?sslmode=require`. Do not use the pooled URL: the per-user locks use session advisory locks and the chat checkpointer needs session state, and both break under PgBouncer transaction pooling.

## 2. Gemini key

Free key at aistudio.google.com.

## 3. Bootstrap admin key

```powershell
py -m auralith.registry create-key admin
```

It prints the secret once and the `admin:<sha256hex>` entry. Keep the secret; the entry goes in `AURALITH_API_KEYS`.

## 4. Render

Dashboard > New > Blueprint > pick this repo (`render.yaml`). Fill the secrets: `DATABASE_URL` (Neon), `GEMINI_API_KEY`, `AURALITH_API_KEYS` (`admin:<sha256hex>`), `AURALITH_ADMINS` (`admin`). After deploy, open `https://<service>.onrender.com/health` and check `"embedding_provider": "gemini"`, no database error and `"auth": true`.

The health check path is `/openapi.json`, not `/health`: `/health` probes the database and would keep Neon awake.

## 5. Streamlit Community Cloud

New app > this repo, branch `main`, main file `ui/streamlit_app.py`, Python 3.12 (Advanced settings). Secrets:

```toml
AURALITH_API_URL = "https://<service>.onrender.com"
```

Users type their API key in the sidebar. Do NOT add `AURALITH_API_KEY` (or any secret other than `AURALITH_API_URL`) to these secrets: string secrets become environment variables, and `AURALITH_API_KEY` would prefill the sidebar for every visitor.

## 6. Keep it awake

Render sleeps a free service after 15 minutes idle (about a minute to wake). UptimeRobot > HTTP(s) monitor on `https://<service>.onrender.com/openapi.json` every 10 minutes. Not `/health`, for the Neon reason above.

## 7. Add users

```powershell
curl.exe -X POST https://<service>.onrender.com/admin/keys -H "X-API-Key: <admin secret>" -H "Content-Type: application/json" -d '{"user":"alice"}'
```

## 8. Smoke test

Open the Streamlit URL, enter a key, upload a PDF in PDF Query, ask a question.

## Guest access (public demo)

With `AURALITH_GUEST_ENABLED=1` (set in `render.yaml`) the Streamlit UI fetches a temporary key from `POST /guest` when the visitor types none, and offers a shared sample café menu.

One-time sample setup (the `sample` user is reserved: its documents are readable by every guest, read-only):

```powershell
curl.exe -X POST https://<service>.onrender.com/admin/keys -H "X-API-Key: <admin secret>" -H "Content-Type: application/json" -d '{\"user\":\"sample\"}'
curl.exe -X POST https://<service>.onrender.com/ingest -H "X-API-Key: <sample key>" -F "file=@samples/cafe-menu.pdf;type=application/pdf"
```

Revoke the sample key afterwards if you like (`DELETE /admin/keys/<id>`); the document stays. Re-ingest after changing the embedder. `samples/make_cafe_menu.py` regenerates the PDF.

- **Limits** (env, per guest): `AURALITH_GUEST_LLM_PER_MIN=5`, `AURALITH_GUEST_LLM_PER_DAY=20`, `AURALITH_GUEST_INGEST_PER_DAY=2`. **Shared by all guests**: `AURALITH_GUEST_POOL_LLM_PER_DAY=200`, `AURALITH_GUEST_POOL_INGEST_PER_DAY=10`, `AURALITH_GUEST_KEYS_PER_DAY=100` new guests. **Per IP**: `AURALITH_GUEST_PER_IP_HOUR=30` new guests. 0 turns a limit off.
- **Hard caps fail closed**: if the usage counter (Postgres) errors, guest requests and `/guest` get 503 instead of being allowed. Named users still fail open.
- **Kill switch**: `AURALITH_GUEST_ENABLED=0` (or unset) makes `/guest` and `/sample` 404 and stops the sample fallback in `/ask`. Existing guest keys keep working for their own documents until they expire.
- **Expiry**: guest keys die after `AURALITH_GUEST_TTL_HOURS` (24). Each `/guest` call lazily sweeps a few expired guest keys, documents and counters. Guest chat checkpoints are not swept.
- **Shared IPs**: Streamlit Cloud makes the server-side call to `/guest`, so every visitor appears to come from the same IP and the per-IP cap is effectively a global cap there; the shared pool is what protects the Gemini quota. The IP is the right-most `X-Forwarded-For` entry (assumes one proxy, Render); the raw header is logged on each `/guest` call so you can verify. Do not set `FORWARDED_ALLOW_IPS`.

## Limits

- **Measured** in Docker with `--cpus=0.1 --memory=512m` on `eval/corpus/attention.pdf` (pgvector Postgres, Gemini generator and embeddings): peak memory 233 MB (cgroup) / 243 MB (process RSS), no OOM. Ingest of the 15-page PDF 147 s (almost all PDF parsing and chunking on 0.1 CPU; the embedding calls took 4 s). /ask: 17 s the first time, 1 s on a repeat. Render's burst CPU may differ.
- **Re-upload after the switch.** Documents ingested under the MiniLM image have vectors in the old embedding space; they stay listed in `GET /documents`, but `/ask` returns 404 for them (no collection for the new embedder) until the same PDF is uploaded again, which re-indexes it under the same `doc_id`. `DELETE /documents/{doc_id}` removes the document and its vectors under every embedder.
- **One shared Gemini key**: the free quota (requests per minute and per day) is shared by all users. Ingest costs one embedding item per chunk (about 30 for a 15-page paper), each /ask one more. When the quota is hit, `/ingest` retries with backoff for about a minute and then returns 503 with `Retry-After: 60`; no partial index is stored.
- **Privacy.** On the Gemini free tier, Google may use submitted content to improve its products. Tell users not to upload private documents, or use a paid-tier key.
- **One heavy request at a time** (`AURALITH_MAX_INFLIGHT=1`, one worker). Others get a 429/503 until it finishes.
- **Image Query does not work**: it needs a local Ollama vision model.
- Neon free storage is small and its compute suspends when idle; the first query after a pause adds a second or two.
