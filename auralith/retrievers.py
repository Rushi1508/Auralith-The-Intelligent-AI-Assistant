"""Retriever assembly.

One function, `build_retriever`, turns a `RetrievalConfig` into a retriever.
Every Phase B/C feature is a flag on the config rather than a separate code
path, which is what lets the sweep treat "baseline" and "hybrid + rerank +
multiquery" as the same kind of object.

Composition order, outermost last:

    dense (Chroma)  ─┐
                     ├─ EnsembleRetriever ─┐
    sparse (BM25)   ─┘   (RRF fusion)      ├─ MultiQuery ─┐
                                           │              ├─ ContextualCompression
                     (or dense alone) ─────┘              │   (cross-encoder rerank)
                                                          └─> final k passages

Order is forced by what each stage needs. Reranking must be outermost: it is
the only stage that sees a joint (query, passage) score, so it should get the
largest, most permissive candidate pool the earlier stages can produce. Query
expansion sits inside it so that passages found by *any* rewritten query still
compete in a single reranked pool.
"""

from __future__ import annotations

import logging

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from auralith.config import RetrievalConfig, Settings
from auralith.providers import get_chat_model, get_cross_encoder

log = logging.getLogger(__name__)


def build_dense_retriever(
    store: Chroma, config: RetrievalConfig, fetch_k: int | None = None
) -> BaseRetriever:
    """Vector-similarity retriever.

    `fetch_k` overrides how many candidates come out. When reranking is on we
    deliberately over-fetch here so the cross-encoder has something to work
    with; the original prototype's bare `.as_retriever()` silently used k=4.
    """
    return store.as_retriever(search_kwargs={"k": fetch_k or config.k})


def build_sparse_retriever(
    chunks: list[Document], config: RetrievalConfig, fetch_k: int | None = None
):
    """BM25 lexical retriever.

    Complements the dense side rather than competing with it. MiniLM maps text
    to a 384-dim semantic space, which is good at paraphrase and bad at exact
    tokens - model names, metric names, numbers, acronyms like "BLEU" or
    "WordPiece". BM25 is the reverse. Questions about a specific reported
    number are exactly where dense-only retrieval tends to fail.
    """
    from langchain_community.retrievers import BM25Retriever

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = fetch_k or config.k
    return retriever


def build_retriever(
    store: Chroma,
    config: RetrievalConfig,
    settings: Settings,
    bm25_chunks: list[Document] | None = None,
) -> BaseRetriever:
    """Assemble the full retriever described by `config`."""
    # Validate before building anything. A sweep can queue dozens of configs,
    # and a misconfiguration should fail immediately with a message naming the
    # problem rather than surfacing as an AttributeError from somewhere deep in
    # the composition several minutes later.
    if config.use_bm25 and bm25_chunks is None:
        raise ValueError(
            "use_bm25=True requires bm25_chunks; pass the output of "
            "auralith.index.load_chunks_for_bm25 for the same config"
        )
    if store is None:
        raise ValueError("build_retriever requires a vector store")

    # When a cross-encoder will re-sort the results, earlier stages fetch a
    # wider pool. Otherwise they fetch exactly what the generator will see.
    fetch_k = config.rerank_fetch_k if config.use_rerank else config.k

    retriever: BaseRetriever = build_dense_retriever(store, config, fetch_k=fetch_k)

    if config.use_bm25:
        assert bm25_chunks is not None  # narrowed by the guard above
        from langchain_classic.retrievers import EnsembleRetriever

        sparse = build_sparse_retriever(bm25_chunks, config, fetch_k=fetch_k)
        # EnsembleRetriever fuses with Reciprocal Rank Fusion, not raw scores.
        # That is the right call here: BM25 scores are unbounded and corpus
        # dependent while cosine similarity is bounded, so the two are not
        # commensurable. RRF only uses rank position, so no normalisation
        # hyperparameter has to be invented.
        retriever = EnsembleRetriever(
            retrievers=[retriever, sparse],
            weights=[config.dense_weight, config.bm25_weight],
        )

    if config.use_multiquery:
        from langchain_classic.retrievers.multi_query import MultiQueryRetriever

        retriever = MultiQueryRetriever.from_llm(
            retriever=retriever,
            llm=get_chat_model(settings),
            include_original=True,
        )

    if config.use_rerank:
        from langchain_classic.retrievers import ContextualCompressionRetriever
        from langchain_classic.retrievers.document_compressors import (
            CrossEncoderReranker,
        )

        reranker = CrossEncoderReranker(
            model=get_cross_encoder(settings), top_n=config.k
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=reranker, base_retriever=retriever
        )

    return retriever


def retrieve(retriever: BaseRetriever, query: str, config: RetrievalConfig):
    """Run retrieval and enforce the k contract.

    Ensemble and multi-query stages both return unions, which can exceed k.
    Trimming here keeps one promise: the generator sees exactly `config.k`
    passages regardless of how they were found. Without this, "hybrid" configs
    would quietly hand the model more context than dense-only ones and any
    comparison between them would be confounded.
    """
    docs = retriever.invoke(query)
    return docs[: config.k]
