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
from pathlib import Path

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
        infos = [sample]
        _show_sample_pdf()
    else:
        new = None
        if uploaded is not None:
            # Every rerun (each question) would re-POST /ingest and spend ingest quota.
            cache = st.session_state.setdefault("pdf_ingested", {})
            new = cache.get(uploaded.file_id) or _ingest(uploaded)
            if new is None:
                return
            cache[uploaded.file_id] = new
        infos = _pick_existing(new)
        if not infos:
            return
    _ask(infos)


SAMPLE_PDF = Path(__file__).resolve().parent.parent / "samples" / "cafe-menu.pdf"


def _show_sample_pdf() -> None:
    """Let visitors read the sample themselves, so they can check the answers."""
    if not SAMPLE_PDF.exists():
        return
    data = SAMPLE_PDF.read_bytes()
    with st.expander("View the menu (the document the answers come from)", expanded=True):
        try:
            st.pdf(data, height=600)
        except Exception:  # noqa: BLE001 - viewer needs the streamlit-pdf add-on
            st.caption("Preview unavailable here; download the menu below.")
        st.download_button("Download the menu (PDF)", data, "cafe-menu.pdf", "application/pdf")


def _pick_existing(new: dict | None = None) -> list[dict]:
    """Choose which of the caller's ingested documents to ask across (or delete one)."""
    try:
        resp = httpx.get(f"{API_URL}/documents", headers=_headers(), timeout=30)
    except httpx.TransportError:
        st.error(_DOWN)
        return []
    if resp.status_code >= 400:
        st.error(_detail(resp))
        return []
    docs = resp.json()
    if not docs:
        st.info("Upload a PDF to begin.")
        return []
    by_id = {d["doc_id"]: d for d in docs}
    sel = st.session_state.get("pdf_sel")
    if sel is None:  # first visit: everything selected
        st.session_state["pdf_sel"] = list(by_id)
    else:  # drop deleted docs, which would make the multiselect raise
        st.session_state["pdf_sel"] = [i for i in sel if i in by_id]
    if new and st.session_state.get("pdf_added") != new["doc_id"]:  # once per upload
        st.session_state["pdf_added"] = new["doc_id"]
        if new["doc_id"] in by_id and new["doc_id"] not in st.session_state["pdf_sel"]:
            st.session_state["pdf_sel"].append(new["doc_id"])
    label = lambda i: f"{by_id[i]['filename']} ({by_id[i]['n_pages']} pages)"  # noqa: E731
    chosen = st.multiselect(
        "Documents to ask across", list(by_id), key="pdf_sel", format_func=label
    )
    # The doc in the upload box is re-ingested on rerun, so it can't be deleted here.
    deletable = [i for i in by_id if not new or i != new["doc_id"]]
    if new:
        st.caption("Clear the upload box to delete the file you just uploaded.")
    if deletable:
        with st.expander("Delete a document"):
            victim = st.selectbox("Document", deletable, format_func=label)
            if st.button("Delete"):
                r = httpx.delete(
                    f"{API_URL}/documents/{victim}", headers=_headers(), timeout=30
                )
                if r.status_code >= 400:
                    st.error(_detail(r))
                    return []
                st.rerun()
    if not chosen:
        st.info("Select at least one document.")
    return [by_id[i] for i in chosen]


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


def _sources(passages: list[dict], expanded: bool = False) -> None:
    # The passages the answer was actually built from. Without this the user
    # cannot audit a claim, which was the prototype's biggest honesty gap.
    with st.expander(f"Sources ({len(passages)} passages)", expanded=expanded):
        for p in passages:
            name = p.get("filename") or p["doc_id"]
            st.markdown(f"**[{p['n']}]** `{name}` page {p['page']}")
            st.text(p["text"])
            st.divider()


def _ask(infos: list[dict]) -> None:
    ids = [i["doc_id"] for i in infos]
    if st.session_state.get("pdf_history_ids") != ids:  # new scope: old turns no longer apply
        st.session_state["pdf_history_ids"] = ids
        st.session_state["pdf_history"] = []
    history = st.session_state.setdefault("pdf_history", [])

    cols = st.columns(4)
    cols[0].metric("Documents", len(infos))
    cols[1].metric("Pages", sum(i["n_pages"] for i in infos))
    cols[2].metric("Chunks", sum(i["n_chunks"] for i in infos))
    cols[3].metric("Passages/query", infos[0]["k"])
    st.caption(f"Retrieval config: {infos[0]['config']}")

    for turn in history:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            st.markdown(turn["answer"])
            _sources(turn["passages"])

    if infos[0].get("owner") == "sample":
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
                json={
                    "doc_ids": ids,
                    "question": question,
                    "history": [
                        {"question": t["question"], "answer": t["answer"]} for t in history[-6:]
                    ],
                },
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
        history.append(
            {"question": question, "answer": answer.get("answer", ""), "passages": passages}
        )
    standalone = answer.get("standalone_question")
    if standalone and standalone != question:
        st.caption(f"Searched for: {standalone}")

    _sources(passages, expanded=abstained)

    st.caption(f"Answered in {events.get('done', {}).get('latency_seconds', 0):.2f}s")
