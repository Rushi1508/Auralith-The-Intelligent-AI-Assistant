"""HTTP service in front of ingest / index / RAG / the agent graph.

`/ask` calls `answer_question` unchanged. It streams NDJSON (passages, then the
full answer, then done), which is not token streaming: the generator call is
one blocking invoke.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, StreamingResponse
from pydantic import BaseModel

from auralith.config import PRODUCTION_CONFIG, Settings
from auralith.graph.build import run_graph, summarise
from auralith.index import build_index, load_chunks_for_bm25
from auralith.ingest import load_pdf
from auralith.providers import ProviderError, check_backend
from auralith.rag import answer_question
from auralith.retrievers import build_retriever

app = FastAPI(title="Auralith")

MAX_UPLOAD_BYTES = int(os.getenv("AURALITH_MAX_UPLOAD_MB", "25")) * 1024 * 1024

# shortcut: in-memory, lost on restart (re-ingest is cheap, Chroma index is on disk), replaced by Postgres in Phase I-3
DOCS: dict[str, dict] = {}


class AskRequest(BaseModel):
    doc_id: str
    question: str


class ChatRequest(BaseModel):
    query: str


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
    return {
        "status": "ok",
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
        with tempfile.TemporaryDirectory() as tmp:
            pdf_path = Path(tmp) / "upload.pdf"
            pdf_path.write_bytes(data)
            docs = load_pdf(pdf_path, doc_id=name)
        index_root = Path(tempfile.gettempdir()) / "auralith_indexes" / doc_id
        store, report = build_index(docs, config, settings, root=index_root)
        bm25 = load_chunks_for_bm25(docs, config) if config.use_bm25 else None
        retriever = build_retriever(store, config, settings, bm25_chunks=bm25)
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
    DOCS[doc_id] = {"retriever": retriever, "info": info}
    return info


@app.post("/ask")
def ask(body: AskRequest) -> StreamingResponse:
    if body.doc_id not in DOCS:
        raise HTTPException(404, "Unknown doc_id; ingest the PDF first.")
    if not body.question.strip():
        raise HTTPException(422, "Question is empty.")
    settings = Settings.from_env()
    _check_backend(settings)
    retriever = DOCS[body.doc_id]["retriever"]

    def gen():
        r = answer_question(body.question, retriever, PRODUCTION_CONFIG, settings)
        passages = [
            {**c, "text": d.page_content[:1200]} for c, d in zip(r.citations, r.passages)
        ]
        yield json.dumps({"type": "passages", "passages": passages}) + "\n"
        yield json.dumps(
            {"type": "answer", "answer": r.answer, "abstained": r.abstained, "error": r.error}
        ) + "\n"
        yield json.dumps({"type": "done", "latency_seconds": r.latency_seconds}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/chat")
def chat(body: ChatRequest) -> dict:
    settings = Settings.from_env()
    _check_backend(settings)
    state = run_graph(body.query, settings)
    return {
        **summarise(state),
        "search_results": state.get("search_results") or [],
        "code": state.get("code"),
        "language": state.get("language"),
    }
