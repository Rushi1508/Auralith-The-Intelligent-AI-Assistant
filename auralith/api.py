"""HTTP service in front of ingest / index / RAG / the agent graph.

`/ask` calls `answer_question` eagerly (inside the in-flight cap), then returns
the result as NDJSON lines (passages, answer, done). It is not token streaming:
the generator call is one blocking invoke.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response, StreamingResponse
import psycopg
from pydantic import BaseModel, Field

from auralith.config import PRODUCTION_CONFIG, Settings
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
from auralith.rag import answer_question
from auralith.registry import (
    delete_doc,
    ensure_schema,
    get_doc,
    list_docs,
    load_keys,
    save_doc,
    user_for_key,
)
from auralith.retrievers import build_retriever, build_sparse_retriever

app = FastAPI(title="Auralith")

MAX_UPLOAD_BYTES = int(os.getenv("AURALITH_MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_QUESTION_CHARS = int(os.getenv("AURALITH_MAX_QUESTION_CHARS", "2000"))
MAX_INFLIGHT = int(os.getenv("AURALITH_MAX_INFLIGHT", "4"))
_SLOTS = threading.BoundedSemaphore(MAX_INFLIGHT)

# Dev mode only (no DATABASE_URL): in-memory registry, lost on restart. With a
# database the registry is the auralith_docs table and _CACHE holds built retrievers.
DOCS: dict[str, dict] = {}
# shortcut: per-process cache, each worker rebuilds once per doc
_CACHE: dict[tuple[str, str], dict] = {}
_CACHE_MAX = 32
# Degradations seen at request time, surfaced by /health.
DEGRADED: dict[str, str] = {}


class AskRequest(BaseModel):
    doc_id: str
    question: Annotated[str, Field(max_length=MAX_QUESTION_CHARS)]


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


def current_user(x_api_key: Annotated[str | None, Header()] = None) -> str:
    """API-key owner; "" (auth off) when AURALITH_API_KEYS is unset."""
    try:
        keys = load_keys()
        user = user_for_key(x_api_key) if keys and x_api_key else None
    except ValueError as exc:  # fail closed, never auth-off
        raise HTTPException(503, "auth misconfigured") from exc
    if not keys:
        return ""
    if user is None:
        raise HTTPException(
            401, "Invalid or missing API key.", headers={"WWW-Authenticate": "ApiKey"}
        )
    return user


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
        coll = collection_name(doc_id, PRODUCTION_CONFIG)
        store = open_pgvector(coll, settings)
        chunks = load_pg_chunks(coll, settings)
        if not chunks:  # registry row without vectors: treat as unknown
            return None
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "Database unavailable") from exc
    entry = {
        "retriever": build_retriever(
            store, PRODUCTION_CONFIG, settings,
            bm25_chunks=chunks if PRODUCTION_CONFIG.use_bm25 else None,
        ),
        "sparse": build_sparse_retriever(chunks, PRODUCTION_CONFIG),
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
    settings, config = Settings.from_env(), PRODUCTION_CONFIG
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
            coll = collection_name(doc_id, PRODUCTION_CONFIG)
            try:
                found = delete_doc(settings.database_url, user, doc_id, coll)
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(503, "Database unavailable") from exc
            _CACHE.pop((user, doc_id), None)
    if not found:
        raise HTTPException(404, "Unknown doc_id.")
    return Response(status_code=204)


@app.post("/ask")
def ask(
    body: AskRequest, user: Annotated[str, Depends(current_user)]
) -> StreamingResponse:
    settings = Settings.from_env()
    doc = _get_doc(body.doc_id, user, settings)
    if doc is None:
        raise HTTPException(404, "Unknown doc_id; ingest the PDF first.")
    if not body.question.strip():
        raise HTTPException(422, "Question is empty.")
    _check_backend(settings)
    with _slot():
        r = answer_question(
            body.question, doc["retriever"], PRODUCTION_CONFIG, settings,
            fallback=doc["sparse"],
        )
    degraded = "degraded" in r.meta
    if degraded:
        DEGRADED["retrieval"] = str(r.meta["degraded"])
    elif r.passages:
        DEGRADED.pop("retrieval", None)
    passages = [
        {**c, "text": d.page_content[:1200]} for c, d in zip(r.citations, r.passages)
    ]
    lines = [
        json.dumps({"type": "passages", "passages": passages}) + "\n",
        json.dumps(
            {
                "type": "answer",
                "answer": r.answer,
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
