"""Shared fixtures.

Tests are split by cost so the fast ones stay usable as a feedback loop:

* Pure-function tests (metrics, config validation, judge parsing) need nothing
  and run in milliseconds.
* Tests marked `corpus` need the fetched PDFs.
* Tests marked `slow` load a real embedding or cross-encoder model.
* Tests marked `ollama` need a running Ollama daemon.

Anything unavailable is skipped with a message saying how to get it, rather
than failing and burying a real regression in environmental noise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.documents import Document

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from auralith.config import CORPUS_DIR, RetrievalConfig  # noqa: E402


def pytest_configure(config: pytest.Config) -> None:
    for marker in (
        "corpus: needs the fetched PDF corpus",
        "slow: loads a real ML model",
        "ollama: needs a running Ollama daemon",
    ):
        config.addinivalue_line("markers", marker)


@pytest.fixture(scope="session")
def corpus_available() -> bool:
    return CORPUS_DIR.exists() and any(CORPUS_DIR.glob("*.pdf"))


@pytest.fixture
def require_corpus(corpus_available: bool) -> None:
    if not corpus_available:
        pytest.skip("Corpus missing. Run: python eval/corpus/fetch_corpus.py")


@pytest.fixture
def sample_docs() -> list[Document]:
    """Synthetic page Documents shaped exactly like PyPDFLoader output.

    Using synthetic pages rather than the real corpus keeps chunking tests
    deterministic and independent of arXiv.
    """
    return [
        Document(
            page_content=(
                "The encoder is composed of a stack of N = 6 identical layers.\n\n"
                "Each layer has two sub-layers: multi-head self-attention and a "
                "position-wise fully connected feed-forward network.\n\n"
                "We employ a residual connection around each sub-layer, followed "
                "by layer normalization."
            ),
            metadata={"doc_id": "attention", "page": 3, "source_path": "attention.pdf"},
        ),
        Document(
            page_content=(
                "We use WordPiece embeddings with a 30,000 token vocabulary.\n\n"
                "The first token of every sequence is always a special "
                "classification token [CLS]."
            ),
            metadata={"doc_id": "bert", "page": 4, "source_path": "bert.pdf"},
        ),
    ]


@pytest.fixture
def base_config() -> RetrievalConfig:
    return RetrievalConfig(
        name="test", splitter="recursive", chunk_size=1000, chunk_overlap=150, k=4
    )


@pytest.fixture
def retrieved_docs() -> list[Document]:
    """Four retrieved chunks; the relevant one (attention p3) sits at index 2."""
    return [
        Document(page_content="x", metadata={"doc_id": "bert", "page": 7}),
        Document(page_content="x", metadata={"doc_id": "rag", "page": 2}),
        Document(page_content="x", metadata={"doc_id": "attention", "page": 3}),
        Document(page_content="x", metadata={"doc_id": "attention", "page": 9}),
    ]
