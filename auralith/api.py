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

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
import psycopg
from pydantic import BaseModel, Field

from auralith.config import PRODUCTION_CONFIG, Settings
from auralith.graph.build import run_graph, summarise
from auralith.index import build_index, load_chunks_for_bm25
from auralith.ingest import load_pdf
from auralith.providers import ProviderError, check_backend
from auralith.rag import answer_question
from auralith.retrievers import build_retriever, build_sparse_retriever

app = FastAPI(title="Auralith")

MAX_UPLOAD_BYTES = int(os.getenv("AURALITH_MAX_UPLOAD_MB", "25")) * 1024 * 1024
MAX_QUESTION_CHARS = int(os.getenv("AURALITH_MAX_QUESTION_CHARS", "2000"))
MAX_INFLIGHT = int(os.getenv("AURALITH_MAX_INFLIGHT", "4"))
_SLOTS = threading.BoundedSemaphore(MAX_INFLIGHT)

# shortcut: in-memory registry, lost on restart; vectors persist (Chroma/pgvector) so
# re-ingest is a cache hit, but the registry itself moves to Postgres in a later phase
DOCS: dict[str, dict] = {}
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
    # shortcut: per-process cap, fine for the single uvicorn worker the Dockerfile
    # runs; use a shared limiter if workers > 1
    if not _SLOTS.acquire(blocking=False):
        raise HTTPException(
            503, "Server busy; retry shortly.", headers={"Retry-After": "5"}
        )
    try:
        yield
    finally:
        _SLOTS.release()


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
    return {
        "status": "degraded" if checks else "ok",
        "checks": checks,
        "provider": s.provider,
        "generator_provider": s.generator_provider,
        "generator_model": s.generator_model,
        "embedding_model": s.embedding_model,
    }


@app.post("/ingest")
def ingest(file: UploadFile) -> dict:
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

    doc_id = hashlib.sha256(data).hexdigest()[:12]
    if doc_id in DOCS:
        return DOCS[doc_id]["info"]

    settings, config = Settings.from_env(), PRODUCTION_CONFIG
    # The client filename is only a display label for citations, never a path.
    name = Path(file.filename).stem
    try:
        with _slot(), tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "upload.pdf"
            pdf_path.write_bytes(data)
            docs = load_pdf(pdf_path, doc_id=name)
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
    }
    DOCS[doc_id] = {"retriever": retriever, "sparse": sparse, "info": info}
    return info


@app.post("/ask")
def ask(body: AskRequest) -> StreamingResponse:
    if body.doc_id not in DOCS:
        raise HTTPException(404, "Unknown doc_id; ingest the PDF first.")
    if not body.question.strip():
        raise HTTPException(422, "Question is empty.")
    settings = Settings.from_env()
    _check_backend(settings)
    doc = DOCS[body.doc_id]
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
def chat(body: ChatRequest) -> dict:
    settings = Settings.from_env()
    _check_backend(settings)
    with _slot():
        state = run_graph(body.query, settings, thread_id=body.thread_id)
    return {
        **summarise(state),
        "thread_id": body.thread_id,
        "search_results": state.get("search_results") or [],
        "code": state.get("code"),
        "language": state.get("language"),
    }
