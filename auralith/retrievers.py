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


DECOMPOSE_PROMPT = """Split the question into independent, self-contained sub-questions, one per fact it asks for.
- If it asks for only one thing, return it unchanged as a single item.
- Each sub-question must name the paper/model/dataset it is about, so it can be searched alone.
- At most {n} sub-questions.
Respond with ONLY a JSON array of strings.

Question: {question}"""


class DecomposeRetriever(BaseRetriever):
    """Retrieve per sub-question, then interleave.

    A cross-encoder scores (query, passage) against the *whole* question, so on
    a multi-part question it keeps passages for the easiest part and drops the
    rest. Retrieving + reranking each part separately, with the original
    question as one more query, keeps evidence for every part in the final k.
    """

    inner: BaseRetriever
    llm: object
    max_parts: int = 3
    k: int = 6

    def _split(self, question: str) -> list[str]:
        import json
        import re

        try:
            raw = self.llm.invoke(
                DECOMPOSE_PROMPT.format(n=self.max_parts, question=question)
            )
            text = raw.content if hasattr(raw, "content") else str(raw)
            m = re.search(r"\[.*\]", text, re.DOTALL)
            parts = [p.strip() for p in json.loads(m.group(0)) if str(p).strip()]
        except Exception as exc:  # noqa: BLE001 - degrade to plain retrieval
            log.warning("Decompose failed (%s); using the original question", exc)
            return [question]
        return parts[: self.max_parts] or [question]

    def _get_relevant_documents(self, query, *, run_manager=None):
        queries = [query] + [q for q in self._split(query) if q != query]
        lists = [self.inner.invoke(q) for q in queries]
        merged, seen = [], set()
        for rank in range(max(len(l) for l in lists)):
            for docs in lists:
                if rank < len(docs):
                    d = docs[rank]
                    key = (d.metadata.get("doc_id"), d.metadata.get("page"), d.page_content)
                    if key not in seen:
                        seen.add(key)
                        merged.append(d)
        return merged[: self.k]


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

    if config.use_decompose:
        retriever = DecomposeRetriever(
            inner=retriever,
            llm=get_chat_model(settings),
            max_parts=config.decompose_max,
            k=config.k,
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
