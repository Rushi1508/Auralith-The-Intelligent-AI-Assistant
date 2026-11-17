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

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

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


def index_path(index_key: str, root: Path = INDEX_DIR) -> Path:
    return root / index_key


def build_index(
    docs: list[Document],
    config: RetrievalConfig,
    settings: Settings,
    root: Path = INDEX_DIR,
    force: bool = False,
) -> tuple[Chroma, IndexBuildReport]:
    """Build (or reuse) the persisted index for this config's chunking scheme.

    Returns the store plus a report, so callers can record whether the work was
    real or served from disk.
    """
    path = index_path(config.index_key, root)
    embeddings = get_embeddings(settings)

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
