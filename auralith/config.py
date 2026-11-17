"""Central configuration for Auralith.

Two ideas live here and nothing else:

1. `Settings` - process-wide settings read from the environment. This exists
   because the original prototype assigned `os.environ["OPENAI_API_KEY"] = ""`
   inline in two modules, which both hardcoded a secret and clobbered any key
   the user had legitimately exported.

2. `RetrievalConfig` - a frozen, hashable description of one retrieval
   pipeline. Every stage (chunking, indexing, retriever assembly, evaluation)
   is a pure function of this object. That is what makes the Phase B/C sweeps
   possible: a sweep is just a list of `RetrievalConfig`s, and a result file is
   keyed by `config.slug`, so a number can always be traced back to the exact
   settings that produced it.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = REPO_ROOT / "eval" / "corpus"
INDEX_DIR = REPO_ROOT / ".chroma"
RESULTS_DIR = REPO_ROOT / "results"

SplitterName = Literal["character", "recursive"]
Provider = Literal["ollama", "openai"]


@dataclass(frozen=True)
class Settings:
    """Runtime settings, all overridable by environment variable."""

    provider: Provider = "ollama"
    generator_model: str = "llama3.2:3b"
    judge_model: str = "qwen2.5:7b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str | None = None
    # Judge and router both need determinism for a reproducible eval.
    temperature: float = 0.0
    request_timeout: int = 120

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("AURALITH_PROVIDER", "ollama"),  # type: ignore[arg-type]
            generator_model=os.getenv("AURALITH_GENERATOR_MODEL", "llama3.2:3b"),
            judge_model=os.getenv("AURALITH_JUDGE_MODEL", "qwen2.5:7b"),
            embedding_model=os.getenv(
                "AURALITH_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            cross_encoder_model=os.getenv(
                "AURALITH_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            temperature=float(os.getenv("AURALITH_TEMPERATURE", "0.0")),
            request_timeout=int(os.getenv("AURALITH_TIMEOUT", "120")),
        )


@dataclass(frozen=True)
class RetrievalConfig:
    """One fully-specified retrieval pipeline.

    Frozen so it can be used as a dict key and so a sweep cannot accidentally
    mutate a shared config between runs.
    """

    name: str = "baseline"

    # --- chunking -------------------------------------------------------
    splitter: SplitterName = "character"
    chunk_size: int = 7500
    chunk_overlap: int = 100

    # --- dense retrieval ------------------------------------------------
    k: int = 4
    """Number of passages handed to the generator."""

    # --- hybrid (Phase C) -----------------------------------------------
    use_bm25: bool = False
    dense_weight: float = 0.5
    """Weight of the dense retriever in the ensemble; BM25 gets 1 - this."""

    # --- reranking (Phase C) --------------------------------------------
    use_rerank: bool = False
    rerank_fetch_k: int = 20
    """Candidates pulled before the cross-encoder narrows them to `k`."""

    # --- query expansion (Phase C) --------------------------------------
    use_multiquery: bool = False
    multiquery_n: int = 3

    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )
        if not 0.0 <= self.dense_weight <= 1.0:
            raise ValueError(f"dense_weight must be in [0, 1], got {self.dense_weight}")
        if self.k < 1:
            raise ValueError(f"k must be >= 1, got {self.k}")
        if self.use_rerank and self.rerank_fetch_k < self.k:
            raise ValueError(
                f"rerank_fetch_k ({self.rerank_fetch_k}) must be >= k ({self.k}); "
                "reranking cannot invent passages it was never given"
            )

    @property
    def bm25_weight(self) -> float:
        return round(1.0 - self.dense_weight, 4)

    @property
    def index_key(self) -> str:
        """Identifies the *vector index*, not the full pipeline.

        Only chunking affects what is stored in Chroma. k, reranking, hybrid
        weighting and query expansion all happen at query time, so configs that
        differ only in those fields share one index. This is what makes a 30+
        config sweep cheap: we embed each chunking scheme once, not once per
        config.
        """
        return f"{self.splitter}_cs{self.chunk_size}_co{self.chunk_overlap}"

    @property
    def slug(self) -> str:
        """Stable, filesystem-safe identifier for this exact pipeline."""
        parts = [self.index_key, f"k{self.k}"]
        if self.use_bm25:
            parts.append(f"bm25w{self.bm25_weight:.2f}")
        if self.use_rerank:
            parts.append(f"rr{self.rerank_fetch_k}")
        if self.use_multiquery:
            parts.append(f"mq{self.multiquery_n}")
        return "__".join(parts)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.slug.encode()).hexdigest()[:10]

    def with_(self, **changes: object) -> "RetrievalConfig":
        """Return a copy with fields replaced. Sweeps are built from this."""
        return replace(self, **changes)  # type: ignore[arg-type]

    def describe(self) -> str:
        bits = [f"{self.splitter} {self.chunk_size}/{self.chunk_overlap}", f"k={self.k}"]
        bits.append(
            f"hybrid(dense={self.dense_weight:.2f}/bm25={self.bm25_weight:.2f})"
            if self.use_bm25
            else "dense-only"
        )
        if self.use_rerank:
            bits.append(f"rerank {self.rerank_fetch_k}->{self.k}")
        if self.use_multiquery:
            bits.append(f"multiquery x{self.multiquery_n}")
        return ", ".join(bits)


# The configuration the inherited prototype actually ran, transcribed field for
# field from the original pdf_module.py so the Phase A baseline measures the
# real starting point rather than a convenient strawman:
#   CharacterTextSplitter.from_tiktoken_encoder(chunk_size=7500, chunk_overlap=100)
#   Chroma.from_documents(...); vectorstore.as_retriever()   # k defaults to 4
BASELINE_CONFIG = RetrievalConfig(
    name="baseline",
    splitter="character",
    chunk_size=7500,
    chunk_overlap=100,
    k=4,
    notes="Original prototype settings, transcribed from pdf_module.py.",
    tags=("phase-a",),
)

# The configuration the Streamlit app serves.
#
# This is deliberately a separate constant from BASELINE_CONFIG, and it changes
# only when a sweep produces evidence for the change. Each edit to it should
# cite the results file that justified it, so `git log -p auralith/config.py`
# reads as the decision history for the pipeline.
#
# Phase A: identical to the baseline. Nothing has been measured yet, so there is
# no evidence on which to choose anything else.
PRODUCTION_CONFIG = BASELINE_CONFIG.with_(
    name="production",
    notes="Phase A: unchanged from baseline; no sweep evidence yet.",
    tags=("production",),
)
