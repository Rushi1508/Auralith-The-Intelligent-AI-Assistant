"""Image question-answering view.

The inherited module passed Streamlit's `UploadedFile` object straight into
`ollama.chat(..., images=[uploaded_image])`, which could not have worked, and
hardcoded the prompt to "Describe this image:" with no question field.

It now posts the image bytes and the user's question to the API's `/image`
endpoint, so it works on the deployed site (Gemini) and locally (an Ollama
vision model), behind the same auth and rate limits as the other modes.
"""

from __future__ import annotations

import httpx
import streamlit as st

from ui.pdf_view import _DOWN, API_URL, _detail, _headers


def start_image_module() -> None:
    st.title("Auralith")
    st.caption("Image Query Mode")
    st.caption(
        "Images are sent to the configured model (Gemini on the public demo; Google may use "
        "free-tier content to improve its products), so don't upload anything private."
    )

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    st.image(uploaded, caption=uploaded.name, use_container_width=True)
    question = st.text_input("Question", value="Describe this image in detail.")
    if not st.button("Ask", type="primary"):
        return

    with st.spinner("Looking at the image..."):
        try:
            resp = httpx.post(
                f"{API_URL}/image",
                files={"file": (uploaded.name, uploaded.getvalue(), uploaded.type)},
                data={"question": question},
                headers=_headers(),
                timeout=None,
            )
        except httpx.TransportError:
            st.error(_DOWN)
            return
    if resp.status_code >= 400:
        st.error(_detail(resp))
        return
    body = resp.json()
    st.markdown(body["answer"])
    st.caption(f"Answered in {body['latency_seconds']:.2f}s")
