"""Tests for chunking and configuration.

The metadata tests are the important ones. Retrieval scoring depends entirely on
every chunk carrying an accurate (doc_id, page) pair - if a splitter dropped or
corrupted page metadata, recall@k would silently collapse toward zero and look
like a retrieval regression rather than a plumbing bug.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from auralith.config import BASELINE_CONFIG, RetrievalConfig
from auralith.ingest import build_splitter, chunk_documents, chunk_stats


class TestRetrievalConfigValidation:
    def test_rejects_overlap_at_or_above_chunk_size(self):
        with pytest.raises(ValueError, match="chunk_overlap"):
            RetrievalConfig(chunk_size=500, chunk_overlap=500)

    def test_rejects_out_of_range_dense_weight(self):
        with pytest.raises(ValueError, match="dense_weight"):
            RetrievalConfig(dense_weight=1.5)

    def test_rejects_k_below_one(self):
        with pytest.raises(ValueError, match="k must be"):
            RetrievalConfig(k=0)

    def test_rejects_rerank_pool_smaller_than_k(self):
        """A reranker cannot return more passages than it was given."""
        with pytest.raises(ValueError, match="rerank_fetch_k"):
            RetrievalConfig(k=10, use_rerank=True, rerank_fetch_k=4)

    def test_bm25_weight_is_complement_of_dense(self):
        assert RetrievalConfig(dense_weight=0.7).bm25_weight == pytest.approx(0.3)

    def test_is_frozen(self):
        with pytest.raises(Exception):
            RetrievalConfig().chunk_size = 999  # type: ignore[misc]

    def test_with_returns_modified_copy(self):
        base = RetrievalConfig(chunk_size=1000)
        assert base.with_(k=8).k == 8
        assert base.k == 4  # original untouched


class TestIndexKeyAndSlug:
    def test_index_key_depends_only_on_chunking(self):
        """Query-time settings must not fork the index.

        This is what lets a 30-config sweep reuse a handful of embedding passes
        instead of re-embedding the corpus for every config.
        """
        a = RetrievalConfig(chunk_size=1000, chunk_overlap=100, k=4)
        b = a.with_(k=10, use_bm25=True, use_rerank=True, rerank_fetch_k=20)
        assert a.index_key == b.index_key

    def test_index_key_changes_with_chunk_size(self):
        a = RetrievalConfig(chunk_size=1000, chunk_overlap=100)
        assert a.index_key != a.with_(chunk_size=500).index_key

    def test_slug_distinguishes_query_time_settings(self):
        a = RetrievalConfig(chunk_size=1000, chunk_overlap=100, k=4)
        assert a.slug != a.with_(k=8).slug
        assert a.slug != a.with_(use_bm25=True).slug

    def test_slug_is_filesystem_safe(self):
        slug = RetrievalConfig(
            chunk_size=1000, chunk_overlap=100, use_bm25=True, use_rerank=True
        ).slug
        assert not set(slug) & set('<>:"/\\|?* ')


class TestSplitterConstruction:
    def test_builds_both_splitter_types(self):
        assert build_splitter(RetrievalConfig(splitter="character")) is not None
        assert build_splitter(RetrievalConfig(splitter="recursive")) is not None

    def test_rejects_unknown_splitter(self):
        cfg = RetrievalConfig()
        object.__setattr__(cfg, "splitter", "nonsense")
        with pytest.raises(ValueError, match="Unknown splitter"):
            build_splitter(cfg)


class TestChunkMetadata:
    def test_preserves_doc_id_and_page(self, sample_docs, base_config):
        chunks = chunk_documents(sample_docs, base_config)
        assert chunks
        for c in chunks:
            assert c.metadata["doc_id"] in {"attention", "bert"}
            assert isinstance(c.metadata["page"], int)

    def test_chunk_never_straddles_a_page(self, sample_docs, base_config):
        """Relies on PyPDFLoader emitting one Document per page.

        This is the assumption that gives every chunk exactly one page label,
        which is what makes page-level relevance scoring valid at all.
        """
        chunks = chunk_documents(sample_docs, base_config)
        pages_by_doc = {(d.metadata["doc_id"], d.metadata["page"]) for d in sample_docs}
        for c in chunks:
            assert (c.metadata["doc_id"], c.metadata["page"]) in pages_by_doc

    def test_assigns_unique_chunk_ids(self, sample_docs, base_config):
        chunks = chunk_documents(sample_docs, base_config)
        ids = [c.metadata["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))

    def test_identical_text_gets_distinct_ids(self, base_config):
        """Repeated headers and footers must not collide into one chunk ID."""
        docs = [
            Document(page_content="Same header text.", metadata={"doc_id": "d", "page": 1}),
            Document(page_content="Same header text.", metadata={"doc_id": "d", "page": 2}),
        ]
        ids = [c.metadata["chunk_id"] for c in chunk_documents(docs, base_config)]
        assert len(ids) == len(set(ids))

    def test_chunk_ids_are_stable_across_runs(self, sample_docs, base_config):
        """Content-addressed IDs make re-indexing idempotent."""
        first = [c.metadata["chunk_id"] for c in chunk_documents(sample_docs, base_config)]
        second = [c.metadata["chunk_id"] for c in chunk_documents(sample_docs, base_config)]
        assert first == second

    def test_records_the_chunking_scheme(self, sample_docs, base_config):
        chunks = chunk_documents(sample_docs, base_config)
        assert all(c.metadata["chunk_config"] == base_config.index_key for c in chunks)


class TestChunkSizeBehaviour:
    def test_smaller_chunk_size_yields_more_chunks(self, sample_docs):
        big = chunk_documents(sample_docs, RetrievalConfig(splitter="recursive", chunk_size=1000, chunk_overlap=100))
        small = chunk_documents(sample_docs, RetrievalConfig(splitter="recursive", chunk_size=50, chunk_overlap=5))
        assert len(small) > len(big)

    def test_recursive_splitter_respects_requested_size(self, sample_docs):
        """The property CharacterTextSplitter does not have.

        RecursiveCharacterTextSplitter descends through separators until a piece
        fits, so chunks land at or near the requested token budget.
        """
        cfg = RetrievalConfig(splitter="recursive", chunk_size=40, chunk_overlap=5)
        stats = chunk_stats(chunk_documents(sample_docs, cfg))
        assert stats["max"] <= 40 * 1.5

    def test_baseline_setting_does_not_actually_split(self, sample_docs):
        """Pins the inherited prototype's real behaviour.

        At chunk_size=7500 tokens, pages of a few hundred tokens are never
        split, so 'chunking' degenerates to one chunk per page. This is the
        measured finding that motivates the whole Phase B sweep, so it is
        asserted rather than left as a claim in a README.
        """
        chunks = chunk_documents(sample_docs, BASELINE_CONFIG)
        assert len(chunks) == len(sample_docs)
        assert chunk_stats(chunks)["max"] < BASELINE_CONFIG.chunk_size


class TestChunkStats:
    def test_reports_count_and_bounds(self, sample_docs, base_config):
        stats = chunk_stats(chunk_documents(sample_docs, base_config))
        assert stats["count"] > 0
        assert stats["min"] <= stats["p50"] <= stats["max"]

    def test_empty_input_returns_zeros(self):
        assert chunk_stats([])["count"] == 0
