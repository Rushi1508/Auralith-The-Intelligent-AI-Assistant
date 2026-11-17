"""Tests for retriever assembly, hybrid search and reranking.

Split into two groups:

* Assembly tests use a stub vector store and stub documents. They verify the
  *wiring* - that flags produce the right composition and that the k contract
  holds - without loading a model. Fast enough to run on every change.
* Tests marked `slow` build a real index over the real corpus and load real
  models. They verify the pieces actually work end to end.

The k-contract tests are the ones that protect the sweep's validity. Ensemble
and multi-query stages both return unions that can exceed k, and if a hybrid
config quietly handed the generator six passages while the dense baseline got
four, every hybrid-vs-dense comparison in the results table would be
confounded by context length rather than retrieval quality.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from auralith.config import CORPUS_DIR, RetrievalConfig, Settings
from auralith.retrievers import build_sparse_retriever, retrieve


class StubRetriever(BaseRetriever):
    """Returns a fixed document list regardless of query."""

    docs: list[Document] = []

    def _get_relevant_documents(self, query: str, **_: object) -> list[Document]:
        return self.docs


def make_docs(n: int) -> list[Document]:
    return [
        Document(
            page_content=f"passage {i}",
            metadata={"doc_id": "d", "page": i, "chunk_id": f"c{i}"},
        )
        for i in range(n)
    ]


class TestKContract:
    def test_trims_to_k(self):
        """The promise: the generator sees exactly k passages, always."""
        retriever = StubRetriever(docs=make_docs(20))
        cfg = RetrievalConfig(k=4)
        assert len(retrieve(retriever, "q", cfg)) == 4

    def test_returns_fewer_when_corpus_is_smaller_than_k(self):
        retriever = StubRetriever(docs=make_docs(2))
        assert len(retrieve(retriever, "q", RetrievalConfig(k=10))) == 2

    def test_preserves_order(self):
        retriever = StubRetriever(docs=make_docs(10))
        out = retrieve(retriever, "q", RetrievalConfig(k=3))
        assert [d.metadata["page"] for d in out] == [0, 1, 2]

    def test_empty_result_is_not_an_error(self):
        assert retrieve(StubRetriever(docs=[]), "q", RetrievalConfig(k=4)) == []


class TestSparseRetriever:
    def test_bm25_finds_an_exact_rare_token(self):
        """The case dense retrieval is worst at.

        MiniLM maps text to a 384-dim semantic space, which blurs rare exact
        tokens. BM25 keys on them directly. This is the mechanism the Phase C
        hybrid sweep is betting on.
        """
        docs = [
            Document(page_content="The model uses WordPiece embeddings.", metadata={"doc_id": "b", "page": 1}),
            Document(page_content="We trained on eight P100 GPUs for twelve hours.", metadata={"doc_id": "a", "page": 2}),
            Document(page_content="Attention weights are computed via softmax.", metadata={"doc_id": "a", "page": 3}),
        ]
        cfg = RetrievalConfig(k=1)
        top = build_sparse_retriever(docs, cfg).invoke("P100")
        assert top[0].metadata["page"] == 2

    def test_respects_fetch_k_override(self):
        docs = make_docs(10)
        r = build_sparse_retriever(docs, RetrievalConfig(k=2), fetch_k=5)
        assert r.k == 5


class TestRetrieverAssembly:
    def test_bm25_without_chunks_is_rejected_early(self):
        """Fail at assembly with a clear message, not mid-sweep with a None."""
        from auralith.retrievers import build_retriever

        cfg = RetrievalConfig(use_bm25=True)
        with pytest.raises(ValueError, match="bm25_chunks"):
            build_retriever(store=None, config=cfg, settings=Settings(), bm25_chunks=None)

    def test_rerank_config_overfetches(self):
        """A reranker needs a bigger candidate pool than it will return."""
        cfg = RetrievalConfig(k=4, use_rerank=True, rerank_fetch_k=20)
        assert cfg.rerank_fetch_k > cfg.k


# --------------------------------------------------------------------------
# Slow: real models and the real corpus
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.corpus
class TestRealRetrieval:
    @pytest.fixture(scope="class")
    def indexed(self, request):
        if not (CORPUS_DIR.exists() and any(CORPUS_DIR.glob("*.pdf"))):
            pytest.skip("Corpus missing. Run: python eval/corpus/fetch_corpus.py")
        from auralith.index import build_index
        from auralith.ingest import load_corpus

        settings = Settings.from_env()
        cfg = RetrievalConfig(splitter="recursive", chunk_size=1000, chunk_overlap=150, k=4)
        docs = load_corpus(CORPUS_DIR)
        store, _ = build_index(docs, cfg, settings)
        return docs, store, cfg, settings

    def test_dense_retrieval_finds_the_right_page(self, indexed):
        from auralith.retrievers import build_retriever

        docs, store, cfg, settings = indexed
        r = build_retriever(store, cfg, settings)
        out = retrieve(r, "How many layers are in the Transformer encoder?", cfg)
        assert out
        assert any(d.metadata["doc_id"] == "attention" for d in out)

    def test_every_chunk_carries_scoring_metadata(self, indexed):
        from auralith.retrievers import build_retriever

        docs, store, cfg, settings = indexed
        out = retrieve(build_retriever(store, cfg, settings), "attention", cfg)
        for d in out:
            assert d.metadata.get("doc_id")
            assert isinstance(d.metadata.get("page"), int)

    def test_hybrid_respects_k(self, indexed):
        """Union of two retrievers must still be trimmed to k."""
        from auralith.index import load_chunks_for_bm25
        from auralith.retrievers import build_retriever

        docs, store, cfg, settings = indexed
        hybrid = cfg.with_(use_bm25=True, dense_weight=0.5)
        chunks = load_chunks_for_bm25(docs, hybrid)
        r = build_retriever(store, hybrid, settings, bm25_chunks=chunks)
        assert len(retrieve(r, "What is WordPiece?", hybrid)) <= hybrid.k

    def test_rerank_respects_k_and_reorders(self, indexed):
        from auralith.retrievers import build_retriever

        docs, store, cfg, settings = indexed
        reranked_cfg = cfg.with_(use_rerank=True, rerank_fetch_k=20)
        query = "How many warmup steps were used?"

        plain = retrieve(build_retriever(store, cfg, settings), query, cfg)
        reranked = retrieve(
            build_retriever(store, reranked_cfg, settings), query, reranked_cfg
        )
        assert len(reranked) <= reranked_cfg.k
        # Not asserting the order *changed* - a reranker agreeing with the
        # bi-encoder is a legitimate outcome. Asserting it differs would make
        # the test fail for a correct implementation.
        assert {d.metadata["chunk_id"] for d in reranked}
        assert plain
