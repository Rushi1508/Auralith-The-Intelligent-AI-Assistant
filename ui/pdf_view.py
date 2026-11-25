"""PDF question-answering view.

Three behavioural changes from the inherited module, all of which follow from
the evaluation work rather than from taste:

1. **The index is built once per document, not once per question.** The
   prototype re-parsed, re-chunked and re-embedded the PDF inside the request
   handler on every submit. The API caches on the file's content hash, so
   re-uploading the same document is free and asking a second question is
   instant.

2. **Retrieved passages are shown.** The prototype used them as context and
   discarded them, which meant a user had no way to tell a grounded answer from
   a fabricated one. They are rendered in an expander with document and page.

3. **The pipeline is the configuration the evaluation selected**, owned by the
   API (auralith/api.py) rather than retyped here, so the app and the results
   table cannot drift apart.

This view is an HTTP client of that API: `/ingest` builds (or reuses) the index,
`/ask` streams NDJSON (passages, then the full answer, then done).
"""

from __future__ import annotations

import json
import os

import httpx
import streamlit as st

API_URL = os.getenv("AURALITH_API_URL", "http://localhost:8000")
_DOWN = (
    f"Cannot reach the Auralith API at {API_URL}. "
    "Start it with: uvicorn auralith.api:app --port 8000"
)


def _detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json()["detail"])
    except Exception:  # noqa: BLE001
        return f"API error {resp.status_code}"


def start_pdf_module() -> None:
    st.title("Auralith")
    st.caption("PDF Query Mode")

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is None:
        st.info("Upload a PDF to begin.")
        return

    file_bytes = uploaded.getvalue()
    with st.spinner("Indexing document (first time only)..."):
        try:
            resp = httpx.post(
                f"{API_URL}/ingest",
                files={"file": (uploaded.name, file_bytes, "application/pdf")},
                timeout=None,
            )
        except httpx.TransportError:
            st.error(_DOWN)
            return
    if resp.status_code >= 400:
        st.error(_detail(resp))
        return
    info = resp.json()

    cols = st.columns(4)
    cols[0].metric("Pages", info["n_pages"])
    cols[1].metric("Chunks", info["n_chunks"])
    cols[2].metric("Passages/query", info["k"])
    cols[3].metric(
        "Index build",
        "cached" if info["from_cache"] else f"{info['build_seconds']:.1f}s",
    )
    st.caption(f"Retrieval config: {info['config']}")

    question = st.text_input("Question", placeholder="What does this document say about...")
    if not st.button("Ask", type="primary") or not question.strip():
        return

    events: dict[str, dict] = {}
    with st.spinner("Retrieving and answering..."):
        try:
            with httpx.stream(
                "POST",
                f"{API_URL}/ask",
                json={"doc_id": info["doc_id"], "question": question},
                timeout=None,
            ) as r:
                if r.status_code >= 400:
                    r.read()
                    st.error(_detail(r))
                    return
                for line in r.iter_lines():
                    if line:
                        ev = json.loads(line)
                        events[ev["type"]] = ev
        except httpx.TransportError:
            st.error(_DOWN)
            return

    answer = events.get("answer", {})
    passages = events.get("passages", {}).get("passages", [])
    if answer.get("error"):
        st.error(f"Failed: {answer['error']}")
        return

    abstained = answer.get("abstained", False)
    if abstained:
        st.warning(
            "The model reported that the retrieved passages do not contain the "
            "answer. Shown below is what was retrieved."
        )
    else:
        st.markdown(answer.get("answer", ""))

    # The passages the answer was actually built from. Without this the user
    # cannot audit a claim, which was the prototype's biggest honesty gap.
    with st.expander(f"Sources ({len(passages)} passages)", expanded=abstained):
        for p in passages:
            st.markdown(f"**[{p['n']}]** `{p['doc_id']}` page {p['page']}")
            st.text(p["text"])
            st.divider()

    st.caption(f"Answered in {events.get('done', {}).get('latency_seconds', 0):.2f}s")
