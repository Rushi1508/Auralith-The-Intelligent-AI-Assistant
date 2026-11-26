"""Persisted Chroma vector index.

The inherited prototype called `Chroma.from_documents(...)` inside the request
handler with no `persist_directory`, so every question re-parsed the PDF,
re-chunked it, re-embedded every chunk, and threw the index away on return.

Note carefully what that costs and what it does *not* cost: it is a latency and
spend problem, not a quality problem. The same vectors go in and the same
neighbours come out, so persistence does not move recall@k by itself. It is
measured here as a latency win and reported as one - claiming a retrieval-
quality improvement from caching would be dishonest.

One index is built per *chunking scheme* (`config.index_key`), not per config.
k, hybrid weighting, reranking and query expansion are all query-time concerns,
so a 30-config sweep still only pays for a handful of embedding passes.
"""

from __future__ import annotations

import functools
import logging
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from auralith.config import INDEX_DIR, RetrievalConfig, Settings
from auralith.ingest import chunk_documents, chunk_stats
from auralith.providers import get_embeddings

log = logging.getLogger(__name__)


@dataclass
class IndexBuildReport:
    """What building the index actually cost. Used for the latency table."""

    index_key: str
    n_chunks: int
    build_seconds: float
    from_cache: bool
    stats: dict[str, float]


@functools.lru_cache
def _pg_engine(url: str):
    import sqlalchemy

    # psycopg2 is not installed (psycopg 3 is); one pool per URL.
    url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return sqlalchemy.create_engine(url, pool_pre_ping=True)


def collection_name(root_name: str, config) -> str:
    return f"{root_name}__{config.index_key}"


_PG_LOCK = threading.Lock()  # PGVector() registers tables on shared MetaData; not thread-safe


def open_pgvector(collection, settings, embeddings=None, pre_delete=False):
    from langchain_postgres import PGVector

    embeddings = embeddings or get_embeddings(settings)
    with _PG_LOCK:
        return PGVector(
            embeddings=embeddings,
            collection_name=collection,
            connection=_pg_engine(settings.database_url),
            use_jsonb=True,
            pre_delete_collection=pre_delete,
        )


def load_pg_chunks(collection: str, settings) -> list[Document]:
    """Stored chunks of a collection, for rebuilding BM25 without the PDF."""
    from sqlalchemy import text

    with _pg_engine(settings.database_url).connect() as c:
        rows = c.execute(
            text(
                "SELECT e.document, e.cmetadata FROM langchain_pg_embedding e "
                "JOIN langchain_pg_collection c ON e.collection_id = c.uuid "
                "WHERE c.name = :n ORDER BY e.id"
            ),
            {"n": collection},
        ).all()
    return [Document(page_content=d, metadata=m or {}) for d, m in rows]


def _build_pgvector(docs, config, settings, root, force, embeddings):
    # root namespaces per-upload indexes, as it does for the Chroma directory.
    collection = collection_name(root.name, config)
    chunks = chunk_documents(docs, config)
    # The embedding id is a primary key shared across collections; chunk_id
    # metadata stays unchanged.
    ids = [f"{collection}:{c.metadata['chunk_id']}" for c in chunks]
    store = open_pgvector(collection, settings, embeddings, force)
    if ids and len(store.get_by_ids(ids)) == len(ids):
        log.info("Reusing pgvector collection %s (%d chunks)", collection, len(ids))
        return store, IndexBuildReport(config.index_key, len(ids), 0.0, True, {})

    started = time.perf_counter()
    store.add_documents(chunks, ids=ids)
    elapsed = time.perf_counter() - started
    return store, IndexBuildReport(
        config.index_key, len(chunks), round(elapsed, 2), False, chunk_stats(chunks)
    )


def index_path(index_key: str, root: Path = INDEX_DIR) -> Path:
    return root / index_key


def build_index(
    docs: list[Document],
    config: RetrievalConfig,
    settings: Settings,
    root: Path = INDEX_DIR,
    force: bool = False,
) -> tuple[VectorStore, IndexBuildReport]:
    """Build (or reuse) the persisted index for this config's chunking scheme.

    Returns the store plus a report, so callers can record whether the work was
    real or served from disk.
    """
    path = index_path(config.index_key, root)
    embeddings = get_embeddings(settings)
    if settings.database_url:
        return _build_pgvector(docs, config, settings, root, force, embeddings)

    if force and path.exists():
        shutil.rmtree(path)

    # A populated directory means we have already embedded this scheme.
    if path.exists() and any(path.iterdir()):
        store = Chroma(
            collection_name=config.index_key,
            embedding_function=embeddings,
            persist_directory=str(path),
        )
        n = store._collection.count()
        if n > 0:
            log.info("Reusing index %s (%d chunks)", config.index_key, n)
            return store, IndexBuildReport(
                index_key=config.index_key,
                n_chunks=n,
                build_seconds=0.0,
                from_cache=True,
                stats={},
            )
        shutil.rmtree(path)

    chunks = chunk_documents(docs, config)
    stats = chunk_stats(chunks)
    path.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    store = Chroma.from_documents(
        documents=chunks,
        collection_name=config.index_key,
        embedding=embeddings,
        persist_directory=str(path),
        ids=[c.metadata["chunk_id"] for c in chunks],
    )
    elapsed = time.perf_counter() - started

    log.info(
        "Built index %s: %d chunks in %.1fs (mean %.0f tokens/chunk)",
        config.index_key,
        len(chunks),
        elapsed,
        stats["mean"],
    )
    return store, IndexBuildReport(
        index_key=config.index_key,
        n_chunks=len(chunks),
        build_seconds=round(elapsed, 2),
        from_cache=False,
        stats=stats,
    )


def load_chunks_for_bm25(
    docs: list[Document], config: RetrievalConfig
) -> list[Document]:
    """Chunks for the sparse retriever.

    BM25Retriever is in-memory and holds its own copy of the corpus, so it
    cannot read from Chroma. It must be fed chunks produced by the *same*
    splitter settings, or the dense and sparse halves of the ensemble would be
    ranking different units and the fused scores would be meaningless.
    """
    return chunk_documents(docs, config)
