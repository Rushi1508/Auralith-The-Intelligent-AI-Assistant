"""PDF loading and chunking.

The one non-obvious requirement here is that every chunk must carry enough
metadata to decide, later and automatically, whether it is the *right* chunk
for a given question. Retrieval metrics are impossible without that.

`PyPDFLoader` emits one Document per page, and `split_documents` only ever
splits *within* a Document, so a chunk can never straddle a page boundary. That
gives every chunk exactly one (doc_id, page) label for free, which is what
eval/metrics.py uses as its relevance judgement.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TextSplitter,
)

from auralith.config import RetrievalConfig

log = logging.getLogger(__name__)


def build_splitter(config: RetrievalConfig) -> TextSplitter:
    """Construct the splitter described by `config`.

    Both variants measure length in *tokens* via tiktoken rather than
    characters. The original prototype did this too, and it matters: chunk_size
    only means something if its units match the context budget you are
    spending, which is denominated in tokens.

    `character`  - the inherited behaviour. Splits on blank lines only, then
                   greedily merges neighbours up to chunk_size. If a single
                   separator-delimited block exceeds chunk_size it is emitted
                   oversized; CharacterTextSplitter will not break it further.
    `recursive`  - tries "\\n\\n", then "\\n", then " ", then characters,
                   descending only when a piece is still too large. This
                   respects paragraph and sentence boundaries where it can and
                   degrades gracefully where it cannot, so chunks land much
                   closer to the requested size.
    """
    kwargs = {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
    }
    if config.splitter == "character":
        return CharacterTextSplitter.from_tiktoken_encoder(**kwargs)
    if config.splitter == "recursive":
        return RecursiveCharacterTextSplitter.from_tiktoken_encoder(**kwargs)
    raise ValueError(f"Unknown splitter: {config.splitter!r}")


def load_pdf(path: Path, doc_id: str | None = None) -> list[Document]:
    """Load one PDF into per-page Documents with normalised metadata."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python eval/corpus/fetch_corpus.py` first."
        )

    docs = PyPDFLoader(str(path)).load()
    resolved_id = doc_id or path.stem

    for d in docs:
        # PyPDFLoader pages are 0-indexed; humans and the eval set are not.
        raw_page = d.metadata.get("page", 0)
        d.metadata.update(
            {
                "doc_id": resolved_id,
                "source_path": str(path),
                "page": int(raw_page) + 1,
            }
        )

    empty = sum(1 for d in docs if not d.page_content.strip())
    if empty:
        log.warning(
            "%s: %d/%d pages produced no text (scanned or image-only?)",
            path.name,
            empty,
            len(docs),
        )
    return docs


def load_corpus(corpus_dir: Path) -> list[Document]:
    """Load every PDF in a directory, sorted for deterministic ordering."""
    pdfs = sorted(corpus_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs in {corpus_dir}. Run `python eval/corpus/fetch_corpus.py`."
        )
    docs: list[Document] = []
    for pdf in pdfs:
        docs.extend(load_pdf(pdf))
    log.info("Loaded %d pages from %d PDFs", len(docs), len(pdfs))
    return docs


def chunk_documents(
    docs: list[Document], config: RetrievalConfig
) -> list[Document]:
    """Split page Documents into retrieval chunks.

    Each chunk gets a content-addressed `chunk_id`. Stable IDs let Chroma
    upserts be idempotent (re-indexing the same corpus does not duplicate
    rows) and let eval results reference a specific chunk across runs.
    """
    splitter = build_splitter(config)
    chunks = splitter.split_documents(docs)

    seen: dict[str, int] = {}
    for i, c in enumerate(chunks):
        # Reading order: passages reach the model by relevance, so "the 3rd job" needs this.
        c.metadata["chunk_index"] = i
        digest = hashlib.sha256(c.page_content.encode("utf-8")).hexdigest()[:16]
        # Identical text on two pages is common (headers, footers). Disambiguate
        # so genuinely distinct chunks never collide on ID.
        n = seen.get(digest, 0)
        seen[digest] = n + 1
        c.metadata["chunk_id"] = digest if n == 0 else f"{digest}-{n}"
        c.metadata["chunk_config"] = config.index_key

    log.info(
        "Chunked %d pages -> %d chunks (%s, size=%d, overlap=%d)",
        len(docs),
        len(chunks),
        config.splitter,
        config.chunk_size,
        config.chunk_overlap,
    )
    return chunks


def chunk_stats(chunks: list[Document]) -> dict[str, float]:
    """Token-length statistics, used to sanity-check a chunking scheme.

    A scheme whose mean chunk length sits far below the requested chunk_size is
    not actually chunking - it is just returning whatever units the loader
    produced. That is exactly what the inherited 7500-token setting does, and
    the number is worth showing rather than asserting.
    """
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    lengths = [len(enc.encode(c.page_content)) for c in chunks]
    if not lengths:
        return {"count": 0, "mean": 0.0, "min": 0.0, "max": 0.0, "p50": 0.0}
    ordered = sorted(lengths)
    return {
        "count": float(len(lengths)),
        "mean": round(sum(lengths) / len(lengths), 1),
        "min": float(ordered[0]),
        "max": float(ordered[-1]),
        "p50": float(ordered[len(ordered) // 2]),
    }
