"""PDF question-answering view.

Three behavioural changes from the inherited module, all of which follow from
the evaluation work rather than from taste:

1. **The index is built once per document, not once per question.** The
   prototype re-parsed, re-chunked and re-embedded the PDF inside the request
   handler on every submit. Cached on the file's content hash, so re-uploading
   the same document is free and asking a second question is instant.

2. **Retrieved passages are shown.** The prototype used them as context and
   discarded them, which meant a user had no way to tell a grounded answer from
   a fabricated one. They are rendered in an expander with document and page.

3. **The pipeline is the configuration the evaluation selected**, imported from
   auralith.config rather than retyped here, so the app and the results table
   cannot drift apart.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from auralith.config import PRODUCTION_CONFIG, Settings
from auralith.index import build_index, load_chunks_for_bm25
from auralith.ingest import load_pdf
from auralith.providers import ProviderError, check_backend
from auralith.rag import answer_question
from auralith.retrievers import build_retriever


@st.cache_resource(show_spinner=False)
def _prepare(file_bytes: bytes, filename: str, _settings: Settings):
    """Build the retriever for one uploaded PDF.

    Keyed on the file's bytes, so the expensive work happens once per distinct
    document rather than once per question. The leading underscore on
    `_settings` tells Streamlit not to try to hash it.
    """
    config = PRODUCTION_CONFIG
    tmp_dir = Path(tempfile.mkdtemp())
    pdf_path = tmp_dir / filename
    pdf_path.write_bytes(file_bytes)

    doc_id = Path(filename).stem
    docs = load_pdf(pdf_path, doc_id=doc_id)

    digest = hashlib.sha256(file_bytes).hexdigest()[:12]
    index_root = Path(tempfile.gettempdir()) / "auralith_indexes" / digest
    store, report = build_index(docs, config, _settings, root=index_root)

    bm25_chunks = load_chunks_for_bm25(docs, config) if config.use_bm25 else None
    retriever = build_retriever(store, config, _settings, bm25_chunks=bm25_chunks)
    return retriever, config, report, len(docs)


def start_pdf_module() -> None:
    st.title("Auralith")
    st.caption("PDF Query Mode")

    settings = Settings.from_env()
    try:
        check_backend(settings)
    except ProviderError as exc:
        st.error(str(exc))
        return

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is None:
        st.info("Upload a PDF to begin.")
        return

    file_bytes = uploaded.getvalue()
    with st.spinner("Indexing document (first time only)..."):
        try:
            retriever, config, report, n_pages = _prepare(
                file_bytes, uploaded.name, settings
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not index that PDF: {type(exc).__name__}: {exc}")
            return

    cols = st.columns(4)
    cols[0].metric("Pages", n_pages)
    cols[1].metric("Chunks", report.n_chunks)
    cols[2].metric("Passages/query", config.k)
    cols[3].metric(
        "Index build", "cached" if report.from_cache else f"{report.build_seconds:.1f}s"
    )
    st.caption(f"Retrieval config: {config.describe()}")

    question = st.text_input("Question", placeholder="What does this document say about...")
    if not st.button("Ask", type="primary") or not question.strip():
        return

    with st.spinner("Retrieving and answering..."):
        result = answer_question(question, retriever, config, settings)

    if result.error:
        st.error(f"Failed: {result.error}")
        return

    if result.abstained:
        st.warning(
            "The model reported that the retrieved passages do not contain the "
            "answer. Shown below is what was retrieved."
        )
    else:
        st.markdown(result.answer)

    # The passages the answer was actually built from. Without this the user
    # cannot audit a claim, which was the prototype's biggest honesty gap.
    with st.expander(f"Sources ({len(result.passages)} passages)", expanded=result.abstained):
        for i, doc in enumerate(result.passages, 1):
            st.markdown(
                f"**[{i}]** `{doc.metadata.get('doc_id')}` "
                f"page {doc.metadata.get('page')}"
            )
            st.text(doc.page_content[:1200])
            st.divider()

    st.caption(f"Answered in {result.latency_seconds:.2f}s")
