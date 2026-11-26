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


def landing_page() -> None:
    st.title("Auralith")
    st.write(
        "A multi-modal document assistant. The PDF pipeline's configuration was "
        "selected by measurement, not by default - see `results/` and the README."
    )

    settings = Settings.from_env()
    st.caption(
        f"Provider: `{settings.provider}` · generator "
        f"`{settings.generator_model}` ({settings.generator_provider}) · "
        f"embeddings `{settings.embedding_model}`"
    )

    st.subheader("Modes")
    for name, blurb in [
        ("Chat", "General requests, routed by a LangGraph state machine to web "
                 "search, code generation, or conversation."),
        ("PDF Query", "Upload a PDF and ask questions. Retrieved passages are "
                      "shown with their source page."),
        ("Image Query", "Ask questions about an image using a local vision model."),
    ]:
        col_button, col_text = st.columns([1, 4])
        if col_button.button(name, use_container_width=True):
            st.session_state.module = name
            st.rerun()
        col_text.caption(blurb)


def main() -> None:
    st.set_page_config(page_title="Auralith", page_icon="A", layout="centered")
    st.session_state.setdefault("module", "Home")
    st.session_state.api_key = st.sidebar.text_input(
        "API key", value=os.getenv("AURALITH_API_KEY", ""), type="password"
    )

    current = st.session_state.module
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
