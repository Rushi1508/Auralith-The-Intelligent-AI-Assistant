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


def _headers() -> dict:
    key = st.session_state.get("api_key")
    return {"X-API-Key": key} if key else {}


def _detail(resp: httpx.Response) -> str:
    try:
        return str(resp.json()["detail"])
    except Exception:  # noqa: BLE001
        return f"API error {resp.status_code}"


def start_pdf_module() -> None:
    st.title("Auralith")
    st.caption("PDF Query Mode")

    key = st.session_state.get("api_key")
    if st.session_state.get("sample_for") != key:  # once per key; None unless guest mode is on
        st.session_state.sample_for = key
        try:
            r = httpx.get(f"{API_URL}/sample", headers=_headers(), timeout=30)
            st.session_state.sample_info = r.json() if r.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            st.session_state.sample_info = None
    sample = st.session_state.sample_info
    use_sample = sample and st.toggle(
        "Try the sample café menu", key="use_sample",
        help="A made-up café menu: drinks, food, prices, opening hours.",
    )

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded is None and use_sample:
        info = sample
    elif uploaded is None:
        info = _pick_existing()
        if info is None:
            return
    else:
        info = _ingest(uploaded)
        if info is None:
            return
    _ask(info)


def _pick_existing() -> dict | None:
    """Choose (or delete) one of the caller's already-ingested documents."""
    try:
        resp = httpx.get(f"{API_URL}/documents", headers=_headers(), timeout=30)
    except httpx.TransportError:
        st.error(_DOWN)
        return None
    if resp.status_code >= 400:
        st.error(_detail(resp))
        return None
    docs = resp.json()
    if not docs:
        st.info("Upload a PDF to begin.")
        return None
    info = st.selectbox(
        "Or choose one of your documents",
        docs,
        format_func=lambda d: f"{d['filename']} ({d['n_pages']} pages)",
    )
    if st.button("Delete this document"):
        r = httpx.delete(
            f"{API_URL}/documents/{info['doc_id']}", headers=_headers(), timeout=30
        )
        if r.status_code >= 400:
            st.error(_detail(r))
            return None
        st.rerun()
    return info


def _ingest(uploaded) -> dict | None:
    file_bytes = uploaded.getvalue()
    with st.spinner("Indexing document (first time only)..."):
        try:
            resp = httpx.post(
                f"{API_URL}/ingest",
                files={"file": (uploaded.name, file_bytes, "application/pdf")},
                headers=_headers(),
                timeout=None,
            )
        except httpx.TransportError:
            st.error(_DOWN)
            return None
    if resp.status_code >= 400:
        st.error(_detail(resp))
        return None
    return resp.json()


def _ask(info: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Pages", info["n_pages"])
    cols[1].metric("Chunks", info["n_chunks"])
    cols[2].metric("Passages/query", info["k"])
    cols[3].metric(
        "Index build",
        "cached" if info["from_cache"] else f"{info['build_seconds']:.1f}s",
    )
    st.caption(f"Retrieval config: {info['config']}")

    if info.get("owner") == "sample":
        for q in (
            "How much is a large latte?",
            "What vegan options are there?",
            "What time do you close on Sunday?",
            "What's the cheapest dessert?",
        ):
            st.button(q, on_click=st.session_state.__setitem__, args=("pdf_question", q))
    question = st.text_input(
        "Question", placeholder="What does this document say about...", key="pdf_question"
    )
    if not st.button("Ask", type="primary") or not question.strip():
        return

    events: dict[str, dict] = {}
    with st.spinner("Retrieving and answering..."):
        try:
            with httpx.stream(
                "POST",
                f"{API_URL}/ask",
                json={"doc_id": info["doc_id"], "question": question},
                headers=_headers(),
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
    if answer.get("degraded"):
        st.caption("Vector search unavailable; answered with keyword (BM25) retrieval.")

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
