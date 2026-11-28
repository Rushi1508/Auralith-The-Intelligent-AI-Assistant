"""Auralith entry point.

    streamlit run main.py

Replaces the original entry point, which could not start at all: it did
`from modules import Functions` while `Functions.py` sat at the repository root
and no `modules/` package existed, so `streamlit run main.py` raised
ModuleNotFoundError before rendering anything.

Layout now:
    auralith/   retrieval, RAG, agent graph, tools - no Streamlit imports
    ui/         Streamlit views - no model or retrieval logic
    eval/       evaluation harness
    tests/      pytest suite

The separation is what lets the pipeline be evaluated headlessly. Nothing in
auralith/ imports streamlit, so eval/run_eval.py exercises exactly the code the
app runs, without a browser.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

from auralith.config import Settings
from ui.chat_view import start_chat_module
from ui.image_view import start_image_module
from ui.pdf_view import start_pdf_module

MODULES = {
    "Home": None,
    "Chat": start_chat_module,
    "PDF Query": start_pdf_module,
    "Image Query": start_image_module,
}


@st.cache_data(ttl=300)
def _backend_caption() -> str:
    """Describe the models the API actually serves, not this UI process's defaults."""
    s = Settings.from_env()
    info = {
        "generator_model": s.generator_model,
        "generator_provider": s.generator_provider,
        "embedding_model": s.embedding_model,
    }
    if url := os.getenv("AURALITH_API_URL"):
        try:
            info = httpx.get(f"{url.rstrip('/')}/health", timeout=60).json()
        except (httpx.HTTPError, ValueError):
            return "API unreachable (it may be waking up - refresh in a minute)."
    return (
        f"Generator `{info.get('generator_model')}` ({info.get('generator_provider')}) · "
        f"embeddings `{info.get('embedding_model')}`"
    )


def landing_page() -> None:
    st.title("Auralith")
    st.write(
        "A multi-modal document assistant. The PDF pipeline's configuration was "
        "selected by measurement, not by default - see `results/` and the README."
    )

    st.caption(_backend_caption())

    st.subheader("Modes")
    for name, blurb in [
        ("Chat", "General requests, routed by a LangGraph state machine to web "
                 "search, code generation, or conversation."),
        ("PDF Query", "Upload a PDF and ask questions. Retrieved passages are "
                      "shown with their source page."),
        ("Image Query", "Upload an image and ask questions about it."),
    ]:
        col_button, col_text = st.columns([1, 4])
        if col_button.button(name, use_container_width=True):
            st.session_state.module = name
            st.rerun()
        col_text.caption(blurb)


def main() -> None:
    st.set_page_config(page_title="Auralith", page_icon="A", layout="centered")
    st.session_state.setdefault("module", "Home")
    typed = st.sidebar.text_input(
        "API key", value=os.getenv("AURALITH_API_KEY", ""), type="password"
    )
    current = st.session_state.module
    # Only when a mode is opened: landing-page visits and bots must not use the guest-key cap.
    if (not typed and current != "Home" and "guest_key" not in st.session_state
            and os.getenv("AURALITH_API_URL")):
        try:  # one attempt per session; 404 when the API has guest mode off
            r = httpx.post(f"{os.environ['AURALITH_API_URL'].rstrip('/')}/guest", timeout=60)
            st.session_state.guest_key = r.json()["key"] if r.status_code == 201 else None
        except (httpx.HTTPError, ValueError, KeyError):
            st.session_state.guest_key = None
    guest = st.session_state.get("guest_key")
    st.session_state.api_key = typed or guest or ""
    if not typed and guest:
        st.sidebar.caption("You're using a temporary guest session (limits apply).")
    elif not typed and current == "Home":
        st.sidebar.caption("No key needed: open a mode to start a guest session.")
    elif not typed:
        st.sidebar.caption("Enter an API key, or refresh to retry a guest session.")

    if current == "Home":
        landing_page()
    else:
        view = MODULES.get(current)
        if view is None:
            st.session_state.module = "Home"
            st.rerun()
        else:
            view()
        if st.button("Back to home"):
            st.session_state.module = "Home"
            st.rerun()


if __name__ == "__main__":
    main()
