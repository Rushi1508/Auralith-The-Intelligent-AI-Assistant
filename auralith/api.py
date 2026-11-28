"""HTTP service in front of ingest / index / RAG / the agent graph.

`/ask` calls `answer_question` eagerly (inside the in-flight cap), then returns
the result as NDJSON lines (passages, answer, done). It is not token streaming:
the generator call is one blocking invoke.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import tempfile
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
import psycopg
from pydantic import BaseModel, Field

from auralith.config import PRODUCTION_CONFIG, Settings, effective_config
from auralith.graph.build import run_graph, summarise
from auralith.index import (
    build_index,
    collection_name,
    load_chunks_for_bm25,
    load_pg_chunks,
    open_pgvector,
)
from auralith.ingest import load_pdf
from auralith.providers import ProviderError, check_backend
from auralith.rag import answer_question, condense_question
from auralith.registry import (
    _USER,
    GUEST_PREFIX,
    GUEST_TTL_S,
    SAMPLE_OWNER,
    admins,
    count_usage,
    create_key,
    db_user_for_key,
    delete_doc,
    ensure_schema,
    get_doc,
    list_docs,
    list_keys,
    load_keys,
    revoke_key,
    save_doc,
    sweep_guests,
    user_for_key,
)
from auralith.retrievers import (
    build_retriever,
    build_sparse_retriever,
    multi_doc_retriever,
)

app = FastAPI(title="Auralith")

logging.basicConfig(
    level=os.getenv("AURALITH_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("auralith.api")
_RID = re.compile(r"[A-Za-z0-9._-]{1,64}")  # client ids are logged: no spaces/newlines

MAX_UPLOAD_BYTES = int(os.getenv("AURALITH_MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_QUESTION_CHARS = int(os.getenv("AURALITH_MAX_QUESTION_CHARS", "2000"))
MAX_ASK_DOCS = int(os.getenv("AURALITH_MAX_ASK_DOCS", "5"))
MAX_INFLIGHT = int(os.getenv("AURALITH_MAX_INFLIGHT", "4"))
_SLOTS = threading.BoundedSemaphore(MAX_INFLIGHT)
# Per-user quotas, shared across workers via Postgres: bucket -> [(limit, window_seconds)]; 0 = off.
LIMITS = {
    "llm": [
        (int(os.getenv("AURALITH_LLM_PER_MIN", "10")), 60),
        (int(os.getenv("AURALITH_LLM_PER_DAY", "300")), 86400),
    ],
    "ingest": [(int(os.getenv("AURALITH_INGEST_PER_DAY", "30")), 86400)],
}
# Guest mode (AURALITH_GUEST_ENABLED=1): per-guest limits plus one shared pool under an owner
# name ("*guests") that the user-name regex can never produce. Pool/IP checks fail closed.
GUEST_LIMITS = {
    "llm": [
        (int(os.getenv("AURALITH_GUEST_LLM_PER_MIN", "5")), 60),
        (int(os.getenv("AURALITH_GUEST_LLM_PER_DAY", "20")), 86400),
    ],
    "ingest": [(int(os.getenv("AURALITH_GUEST_INGEST_PER_DAY", "2")), 86400)],
}
GUEST_POOL = {
    "llm": [(int(os.getenv("AURALITH_GUEST_POOL_LLM_PER_DAY", "200")), 86400)],
    "ingest": [(int(os.getenv("AURALITH_GUEST_POOL_INGEST_PER_DAY", "10")), 86400)],
    "guest": [(int(os.getenv("AURALITH_GUEST_KEYS_PER_DAY", "100")), 86400)],
}
GUEST_PER_IP_HOUR = int(os.getenv("AURALITH_GUEST_PER_IP_HOUR", "30"))
GUEST_POOL_OWNER = "*guests"
# shortcut: dev-mode (no DATABASE_URL) counters are per process and never pruned
_USAGE: dict[tuple[str, str, int], int] = {}
_USAGE_LOCK = threading.Lock()

# Dev mode only (no DATABASE_URL): in-memory registry, lost on restart. With a
# database the registry is the auralith_docs table and _CACHE holds built retrievers.
DOCS: dict[str, dict] = {}
# shortcut: per-process cache, each worker rebuilds once per doc
_CACHE: dict[tuple[str, str], dict] = {}
_CACHE_MAX = 32
# Degradations seen at request time, surfaced by /health.
DEGRADED: dict[str, str] = {}


class Turn(BaseModel):
    # No length cap: a long echoed answer must not 422; it is truncated when used.
    question: str
    answer: str


class AskRequest(BaseModel):
    doc_id: str | None = None
    doc_ids: Annotated[list[str], Field(max_length=MAX_ASK_DOCS)] | None = None
    question: Annotated[str, Field(max_length=MAX_QUESTION_CHARS)]
    history: Annotated[list[Turn], Field(max_length=6)] = []


class KeyRequest(BaseModel):
    user: Annotated[str, Field(pattern=_USER.pattern)]


class ChatRequest(BaseModel):
    query: Annotated[str, Field(max_length=MAX_QUESTION_CHARS)]
    thread_id: Annotated[str, Field(max_length=128)] | None = None


@contextmanager
def _slot() -> Iterator[None]:
    # shortcut: per-worker cap (DEGRADED is per worker too), so the effective cap is
    # WEB_CONCURRENCY x AURALITH_MAX_INFLIGHT; use a shared limiter if that matters
    if not _SLOTS.acquire(blocking=False):
        raise HTTPException(
            503, "Server busy; retry shortly.", headers={"Retry-After": "5"}
        )
    try:
        yield
    finally:
        _SLOTS.release()


@app.middleware("http")
async def request_log(request: Request, call_next):
    """X-Request-ID on every response plus one JSON access-log line per request."""
    rid = request.headers.get("x-request-id", "")
    rid = rid if _RID.fullmatch(rid) else uuid.uuid4().hex[:16]
    request.state.rid = rid
    t0 = time.perf_counter()
    try:
        resp = await call_next(request)
    except Exception:
        log.exception("unhandled error rid=%s path=%s", rid, request.url.path)
        resp = JSONResponse({"detail": "Internal error", "request_id": rid}, status_code=500)
    resp.headers["X-Request-ID"] = rid
    log.info(json.dumps({
        "rid": rid,
        "method": request.method,
        "path": request.url.path,
        "status": resp.status_code,
        "ms": round((time.perf_counter() - t0) * 1000, 1),
        "user": getattr(request.state, "user", None),
    }))
    return resp


def current_user(
    request: Request, x_api_key: Annotated[str | None, Header()] = None
) -> str:
    """API-key owner; "" (auth off) when AURALITH_API_KEYS is unset."""
    try:
        keys = load_keys()
        user = user_for_key(x_api_key) if keys and x_api_key else None
    except ValueError as exc:  # fail closed, never auth-off
        raise HTTPException(503, "auth misconfigured") from exc
    if not keys:
        return ""
    if user is None and x_api_key:
        url = Settings.from_env().database_url
        try:
            user = db_user_for_key(url, x_api_key) if url else None
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(503, "Database unavailable") from exc
    if user is None:
        raise HTTPException(
            401, "Invalid or missing API key.", headers={"WWW-Authenticate": "ApiKey"}
        )
    request.state.user = user
    return user


def _guest_enabled() -> bool:
    return os.getenv("AURALITH_GUEST_ENABLED") == "1"


def _client_ip(request: Request) -> str:
    # shortcut: assumes exactly one appending proxy (Render); verify by logging raw XFF once
    last = request.headers.get("x-forwarded-for", "").rsplit(",", 1)[-1].strip()
    return last or (request.client.host if request.client else "unknown")


def _limit_plan(user: str, bucket: str) -> list[tuple[str, list[tuple[int, int]]]]:
    """(counter owner, rules) pairs to check, per-user first."""
    if user.startswith(GUEST_PREFIX):
        return [(user, GUEST_LIMITS[bucket]), (GUEST_POOL_OWNER, GUEST_POOL[bucket])]
    return [(user, LIMITS[bucket])]


def _rate_limit(settings: Settings, user: str, bucket: str) -> None:
    # Guests fail closed: the hard cap on shared Gemini quota must hold during an outage.
    for owner, rules in _limit_plan(user, bucket):
        _check(settings, owner, bucket, rules, fail_closed=user.startswith(GUEST_PREFIX))


def _check(
    settings: Settings, user: str, bucket: str, rules: list[tuple[int, int]],
    fail_closed: bool = False,
) -> None:
    """429 once the owner passes any rule in the current window (0 = off)."""
    rules = [(lim, sec) for lim, sec in rules if lim > 0]
    if not rules:
        return
    now = time.time()
    windows = [(f"{bucket}:{sec}", sec) for _, sec in rules]
    try:
        if settings.database_url:
            counts = count_usage(settings.database_url, user, windows, now)
        else:
            with _USAGE_LOCK:
                counts = []
                for key, sec in windows:
                    k = (user, key, int(now // sec))
                    _USAGE[k] = _USAGE.get(k, 0) + 1
                    counts.append(_USAGE[k])
    except Exception as exc:  # noqa: BLE001
        if fail_closed:
            log.warning("rate limiter unavailable; refusing guest request", exc_info=True)
            raise HTTPException(
                503, "Rate limiter unavailable.", headers={"Retry-After": "30"}
            ) from exc
        # a limiter outage must not take the API down for named users
        log.warning("rate limiter unavailable; allowing request", exc_info=True)
        return
    for (lim, sec), n in zip(rules, counts):
        if n > lim:
            raise HTTPException(
                429,
                f"Rate limit reached: {lim} {bucket} requests per {sec}s.",
                headers={"Retry-After": str(int(sec - now % sec) + 1)},
            )


def _cache_put(owner: str, doc_id: str, entry: dict) -> None:
    _CACHE[(owner, doc_id)] = entry
    while len(_CACHE) > _CACHE_MAX:
        _CACHE.pop(next(iter(_CACHE)), None)


def _get_doc(doc_id: str, owner: str, settings: Settings) -> dict | None:
    if not settings.database_url:
        doc = DOCS.get(doc_id)
        return doc if doc and doc["info"].get("owner", "") == owner else None
    try:
        # Registry checked even on a cache hit: another worker may have deleted the doc.
        ensure_schema(settings.database_url)
        info = get_doc(settings.database_url, owner, doc_id)
        if info is None:
            _CACHE.pop((owner, doc_id), None)
            return None
        if hit := _CACHE.get((owner, doc_id)):
            return hit
        coll = collection_name(doc_id, PRODUCTION_CONFIG, settings)
        store = open_pgvector(coll, settings)
        chunks = load_pg_chunks(coll, settings)
        if not chunks:  # registry row without vectors: treat as unknown
            return None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "Database unavailable") from exc
    config = effective_config(PRODUCTION_CONFIG, settings)
    entry = {
        "retriever": build_retriever(
            store, config, settings,
            bm25_chunks=chunks if config.use_bm25 else None,
        ),
        "sparse": build_sparse_retriever(chunks, config),
        "info": info,
    }
    _cache_put(owner, doc_id, entry)
    return entry


@contextmanager
def _doc_lock(settings: Settings, doc_id: str) -> Iterator[None]:
    """Serialise concurrent ingests of one doc across threads and workers."""
    if not settings.database_url:
        yield
        return
    try:
        conn = psycopg.connect(settings.database_url, autocommit=True)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "Database unavailable") from exc
    with conn:
        conn.execute("SELECT pg_advisory_lock(hashtext(%s))", (doc_id,))
        try:
            yield
        finally:
            conn.execute("SELECT pg_advisory_unlock(hashtext(%s))", (doc_id,))


def _probe_db(url: str) -> str | None:
    try:
        with psycopg.connect(url):
            pass
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"
    return None


def _check_backend(settings: Settings) -> None:
    try:
        check_backend(settings)
    except ProviderError as exc:
        raise HTTPException(503, str(exc)) from exc


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse("/docs")


@app.get("/health")
def health() -> dict:
    s = Settings.from_env()
    checks = dict(DEGRADED)
    if s.database_url and (err := _probe_db(s.database_url)):
        checks["database"] = err
    try:
        auth = bool(load_keys())
    except ValueError:
        auth = False
        checks["auth"] = "AURALITH_API_KEYS malformed"
    return {
        "status": "degraded" if checks else "ok",
        "checks": checks,
        "provider": s.provider,
        "generator_provider": s.generator_provider,
        "generator_model": s.generator_model,
        "embedding_model": s.embedding_model,
        "ml_backend": s.ml_backend,
        "embedding_provider": s.embedding_provider,
        "rerank_provider": s.rerank_provider,
        "auth": auth,
    }


@app.post("/ingest")
def ingest(file: UploadFile, user: Annotated[str, Depends(current_user)]) -> dict:
    if (
        Path(file.filename or "").suffix.lower() != ".pdf"
        or file.content_type != "application/pdf"
    ):
        raise HTTPException(415, "Only PDF uploads are accepted.")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    if not data.startswith(b"%PDF-"):
        raise HTTPException(415, "File is not a PDF.")

    doc_id = hashlib.sha256(user.encode() + data).hexdigest()[:12]
    settings = Settings.from_env()
    config = effective_config(PRODUCTION_CONFIG, settings)
    _rate_limit(settings, user, "ingest")
    with _doc_lock(settings, doc_id):
        if existing := _get_doc(doc_id, user, settings):
            return existing["info"]
        # The client filename is only a display label for citations, never a path.
        name = Path(file.filename).stem
        try:
            with _slot(), tempfile.TemporaryDirectory() as tmp:
                pdf_path = Path(tmp) / "upload.pdf"
                pdf_path.write_bytes(data)
                docs = load_pdf(pdf_path, doc_id=name)
                if not any(d.page_content.strip() for d in docs):
                    raise HTTPException(
                        422, "PDF has no extractable text (scanned images need OCR first)."
                    )
                index_root = Path(tempfile.gettempdir()) / "auralith_indexes" / doc_id
                store, report = build_index(docs, config, settings, root=index_root)
                chunks = load_chunks_for_bm25(docs, config)
                retriever = build_retriever(
                    store, config, settings, bm25_chunks=chunks if config.use_bm25 else None
                )
                sparse = build_sparse_retriever(chunks, config)
        except HTTPException:
            raise
        except ProviderError as exc:  # embedding API quota/outage: retryable, not a bad PDF
            if not exc.retryable:  # bad key/request: retrying cannot help
                raise HTTPException(502, str(exc)) from exc
            raise HTTPException(503, str(exc), headers={"Retry-After": "60"}) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(422, f"{type(exc).__name__}: {exc}") from exc

        info = {
            "doc_id": doc_id,
            "filename": name,
            "n_pages": len(docs),
            "n_chunks": report.n_chunks,
            "from_cache": report.from_cache,
            "build_seconds": report.build_seconds,
            "k": config.k,
            "config": config.describe(),
            "owner": user,
        }
        entry = {"retriever": retriever, "sparse": sparse, "info": info}
        if settings.database_url:
            try:
                save_doc(settings.database_url, user, info)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(503, "Database unavailable") from exc
            _cache_put(user, doc_id, entry)
        else:
            DOCS[doc_id] = entry
        return info


@app.get("/documents")
def list_documents(user: Annotated[str, Depends(current_user)]) -> list[dict]:
    """The caller's ingested documents, newest first (dev mode: insertion order)."""
    settings = Settings.from_env()
    if not settings.database_url:
        return [d["info"] for d in DOCS.values() if d["info"].get("owner", "") == user]
    try:
        return list_docs(settings.database_url, user)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "Database unavailable") from exc


@app.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, user: Annotated[str, Depends(current_user)]) -> Response:
    settings = Settings.from_env()
    with _doc_lock(settings, doc_id):  # don't race an in-flight ingest of the same doc
        if not settings.database_url:
            doc = DOCS.get(doc_id)
            found = bool(doc and doc["info"].get("owner", "") == user)
            if found:
                del DOCS[doc_id]
        else:
            try:
                found = delete_doc(settings.database_url, user, doc_id)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(503, "Database unavailable") from exc
            _CACHE.pop((user, doc_id), None)
    if not found:
        raise HTTPException(404, "Unknown doc_id.")
    return Response(status_code=204)


def admin_user(user: Annotated[str, Depends(current_user)]) -> str:
    if not user or user not in admins():
        raise HTTPException(403, "Admin only (AURALITH_ADMINS).")
    return user


def _admin_db() -> str:
    url = Settings.from_env().database_url
    if not url:
        raise HTTPException(501, "Runtime keys need DATABASE_URL.")
    return url


@app.post("/admin/keys", status_code=201)
def admin_create_key(body: KeyRequest, _: Annotated[str, Depends(admin_user)]) -> dict:
    """Issue a key for a (new or existing) user. The raw key is shown only in this response."""
    if body.user.startswith(GUEST_PREFIX):
        raise HTTPException(422, f"User names starting with {GUEST_PREFIX!r} are reserved.")
    url = _admin_db()
    try:
        return create_key(url, body.user)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "Database unavailable") from exc


@app.get("/admin/keys")
def admin_list_keys(_: Annotated[str, Depends(admin_user)]) -> list[dict]:
    """Runtime keys (id, user, created_at); env bootstrap keys aren't listed."""
    url = _admin_db()
    try:
        return list_keys(url)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "Database unavailable") from exc


@app.delete("/admin/keys/{key_id}", status_code=204)
def admin_revoke_key(key_id: int, _: Annotated[str, Depends(admin_user)]) -> Response:
    url = _admin_db()
    try:
        found = revoke_key(url, key_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "Database unavailable") from exc
    if not found:
        raise HTTPException(404, "Unknown key id.")
    return Response(status_code=204)


@app.post("/guest", status_code=201)
def guest(request: Request) -> dict:
    """Anonymous, short-lived key for the public demo (off unless AURALITH_GUEST_ENABLED=1)."""
    if not _guest_enabled():
        raise HTTPException(404, "Guest access is disabled.")
    url = _admin_db()
    log.info("guest issuance x-forwarded-for=%r", request.headers.get("x-forwarded-for"))
    try:
        sweep_guests(url)
    except Exception:  # noqa: BLE001  # best effort, never blocks issuance
        log.warning("guest sweep failed", exc_info=True)
    settings = Settings.from_env()
    _check(settings, f"ip:{_client_ip(request)}", "guest", [(GUEST_PER_IP_HOUR, 3600)], True)
    _check(settings, GUEST_POOL_OWNER, "guest", GUEST_POOL["guest"], True)
    try:
        made = create_key(url, f"{GUEST_PREFIX}{secrets.token_hex(6)}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "Database unavailable") from exc
    return {"key": made["key"], "user": made["user"], "expires_in": GUEST_TTL_S}


@app.get("/sample")
def sample(_: Annotated[str, Depends(current_user)]) -> dict:
    """The shared read-only demo document (guest mode only)."""
    docs = list_documents(SAMPLE_OWNER) if _guest_enabled() else []
    if not docs:
        raise HTTPException(404, "No sample document.")
    return docs[0]


@app.post("/ask")
def ask(
    body: AskRequest, user: Annotated[str, Depends(current_user)]
) -> StreamingResponse:
    settings = Settings.from_env()
    ids = list(dict.fromkeys(body.doc_ids or ([body.doc_id] if body.doc_id else [])))
    if not ids:
        raise HTTPException(422, "doc_id or doc_ids is required.")
    docs = []
    for did in ids:
        doc = _get_doc(did, user, settings)
        if doc is None and _guest_enabled() and user != SAMPLE_OWNER:  # read-only shared sample
            doc = _get_doc(did, SAMPLE_OWNER, settings)
        if doc is None:  # never skip silently: the answer would claim a wrong scope
            raise HTTPException(404, "Unknown doc_id; ingest the PDF first.")
        docs.append(doc)
    if not body.question.strip():
        raise HTTPException(422, "Question is empty.")
    _rate_limit(settings, user, "llm")
    history = [
        {"question": t.question[:MAX_QUESTION_CHARS], "answer": t.answer[:MAX_QUESTION_CHARS]}
        for t in body.history
    ]
    if history:
        _rate_limit(settings, user, "llm")  # the condense call spends LLM quota too
    _check_backend(settings)
    config = effective_config(PRODUCTION_CONFIG, settings)
    with _slot():
        standalone = condense_question(
            history, body.question, settings, MAX_QUESTION_CHARS
        )
        r = answer_question(
            standalone,
            multi_doc_retriever([d["retriever"] for d in docs], config, settings),
            config, settings,
            fallback=multi_doc_retriever(
                [d["sparse"] for d in docs], config, settings, rerank=False
            ),
        )
    infos = [d["info"] for d in docs]
    degraded = "degraded" in r.meta
    if degraded:
        DEGRADED["retrieval"] = str(r.meta["degraded"])
    elif r.passages:
        DEGRADED.pop("retrieval", None)
    passages = []
    for c, d in zip(r.citations, r.passages):
        # shortcut: chunk metadata carries only the filename stem, so same-stem files in
        # one ask can't be told apart (source_doc_id None); tag at retrieval to fix.
        hit = [i for i in infos if i["filename"] == c["doc_id"]]
        src = infos[0] if len(infos) == 1 else hit[0] if len(hit) == 1 else None
        passages.append({
            **c, "text": d.page_content[:1200],
            "source_doc_id": src["doc_id"] if src else None,
            "filename": src["filename"] if src else c["doc_id"],
        })
    lines = [
        json.dumps({"type": "passages", "passages": passages}) + "\n",
        json.dumps(
            {
                "type": "answer",
                "answer": r.answer,
                "standalone_question": standalone,
                "abstained": r.abstained,
                "error": r.error,
                "degraded": degraded,
            }
        )
        + "\n",
        json.dumps({"type": "done", "latency_seconds": r.latency_seconds}) + "\n",
    ]
    return StreamingResponse(iter(lines), media_type="application/x-ndjson")


@app.post("/chat")
def chat(body: ChatRequest, user: Annotated[str, Depends(current_user)]) -> dict:
    settings = Settings.from_env()
    _rate_limit(settings, user, "llm")
    _check_backend(settings)
    tid = f"{user}:{body.thread_id}" if user and body.thread_id else body.thread_id
    with _slot():
        state = run_graph(body.query, settings, thread_id=tid)
    return {
        **summarise(state),
        "thread_id": body.thread_id,
        "search_results": state.get("search_results") or [],
        "code": state.get("code"),
        "language": state.get("language"),
    }
