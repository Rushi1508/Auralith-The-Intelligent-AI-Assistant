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
    # Hosted *generator only* (Gemini's OpenAI-compatible endpoint). The judge,
    # embeddings and cross-encoder stay local so v2 numbers remain comparable.
    generator_provider: Literal["ollama", "gemini"] = "ollama"
    gemini_api_key: str | None = None
    # Plain postgresql:// URL; unset keeps Chroma + the in-memory checkpointer.
    database_url: str | None = None
    # Judge and router both need determinism for a reproducible eval.
    temperature: float = 0.0
    request_timeout: int = 120
    max_output_tokens: int = 512
    """Generation cap.

    Applied uniformly to every config, so it cannot bias a comparison. Set from
    the data rather than by feel: the longest reference answer in the eval set
    is two sentences, so 512 tokens is already several times more headroom than
    any correct answer needs. Its real job is to stop a small model from
    rambling for a thousand tokens on an ambiguous prompt and stalling a run -
    on CPU that is the difference between a 30-minute and a 70-minute pass.
    """

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            provider=os.getenv("AURALITH_PROVIDER", "ollama"),  # type: ignore[arg-type]
            generator_model=os.getenv(
                "AURALITH_GENERATOR_MODEL",
                "gemini-3.5-flash-lite"
                if os.getenv("AURALITH_GENERATOR_PROVIDER") == "gemini"
                else "llama3.2:3b",
            ),
            judge_model=os.getenv("AURALITH_JUDGE_MODEL", "qwen2.5:7b"),
            embedding_model=os.getenv(
                "AURALITH_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            cross_encoder_model=os.getenv(
                "AURALITH_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
            ),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
            generator_provider=os.getenv("AURALITH_GENERATOR_PROVIDER", "ollama"),  # type: ignore[arg-type]
            gemini_api_key=os.getenv("GEMINI_API_KEY") or None,
            database_url=os.getenv("DATABASE_URL") or None,
            temperature=float(os.getenv("AURALITH_TEMPERATURE", "0.0")),
            request_timeout=int(os.getenv("AURALITH_TIMEOUT", "120")),
            max_output_tokens=int(os.getenv("AURALITH_MAX_OUTPUT_TOKENS", "512")),
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

    # --- query decomposition (Phase F) ----------------------------------
    use_decompose: bool = False
    decompose_max: int = 3
    """Max sub-questions. Each is retrieved + reranked on its own, then merged."""

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
        if self.use_decompose:
            parts.append(f"dec{self.decompose_max}")
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
        if self.use_decompose:
            bits.append(f"decompose <={self.decompose_max}")
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
# Phase B (results/sweep_chunking.md, results/sweep_k.md):
#
#   chunking, k held at 4        recall@4     MRR
#     character 7500/100 (old)     0.6154   0.5096   <- 50 chunks; never splits
#     recursive 500/50             0.6346   0.5192
#     recursive 500/100            0.8077   0.6763   <- winner
#     recursive 1000/100           0.5000   0.4071   <- worse than doing nothing
#     recursive 1000/200           0.6731   0.5801
#     recursive 1500/150           0.6154   0.5096   <- 50 chunks; never splits
#
#   k, on recursive 500/100      recall@k   hit@k     MRR   prec@k
#     k=4                          0.8077  0.8462  0.6763   0.2692
#     k=6                          0.8846  0.9231  0.6917   0.2308   <- winner
#     k=8                          0.9038  0.9231  0.6917   0.1971
#     k=10                         0.9038  0.9231  0.6917   0.1923
#
# k=6 rather than k=8: hit@k and MRR both reach their ceiling at 6. The extra
# 0.019 recall at k=8 comes from one multi-page question and costs 33% more
# context while dropping precision 0.231 -> 0.197. recall@k rises with k by
# construction, so it is not a like-for-like column across these rows.
#
# Phase C left everything off by default:
#   hybrid BM25 - rejected, +0.0045 MRR (inside noise). results/sweep_hybrid.md
#   reranking   - trade-off, not a win.                  results/sweep_rerank.md
#
# Phase F step 4 (results/comparison_ci.md, "Dense vs reranked retrieval"):
#   reranking 20 -> 6, end to end on the 3B generator
#     hand 19 -> 22-24/28, 3 answers fixed, 0 broken; hit@k 0.923 -> 0.885
#     (bert-006, a wrong refusal in both runs), MRR 0.692 -> 0.705.
#   Switched on: about the same as the 7B generator for one cross-encoder pass.
#   Strict CI [0, +0.25] touches 0, so it is a default, not a claimed gain.
PRODUCTION_CONFIG = RetrievalConfig(
    name="production",
    splitter="recursive",
    chunk_size=500,
    chunk_overlap=100,
    k=6,
    use_rerank=True,
    rerank_fetch_k=20,
    notes="Phase B chunking/k + Phase F rerank. See results/comparison_ci.md.",
    tags=("production", "phase-b", "phase-f"),
)
