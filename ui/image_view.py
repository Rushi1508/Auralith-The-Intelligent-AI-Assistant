"""Image question-answering view.

The inherited module passed Streamlit's `UploadedFile` object straight into
`ollama.chat(..., images=[uploaded_image])`. Ollama expects file paths or raw
bytes, not a file-like wrapper, so that call could not have worked as written.
It also hardcoded the prompt to "Describe this image:" with no question field,
despite the mode being called Image Query.

Both are fixed here: bytes are read out of the upload, and the user can ask an
actual question.
"""

from __future__ import annotations

import streamlit as st

DEFAULT_VISION_MODEL = "llava"


def start_image_module() -> None:
    st.title("Auralith")
    st.caption("Image Query Mode")

    try:
        import ollama
    except ImportError:
        st.error("The `ollama` package is not installed: pip install ollama")
        return

    model = st.text_input("Vision model", value=DEFAULT_VISION_MODEL)

    # Checked up front because the vision model is a separate multi-GB pull from
    # the text generator, and "model not found" is by far the likeliest failure.
    try:
        available = {m.get("model") or m.get("name") for m in ollama.list().get("models", [])}
    except Exception as exc:  # noqa: BLE001
        st.error(f"Cannot reach Ollama: {exc}. Is `ollama serve` running?")
        return

    if model not in available and f"{model}:latest" not in available:
        st.warning(
            f"Model `{model}` is not pulled. Run `ollama pull {model}` first.\n\n"
            f"Available: {', '.join(sorted(a for a in available if a)) or 'none'}"
        )
        return

    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    st.image(uploaded, caption=uploaded.name, use_container_width=True)
    question = st.text_input("Question", value="Describe this image in detail.")

    if not st.button("Ask", type="primary"):
        return

    # getvalue() returns bytes, which is what ollama actually accepts. Passing
    # the UploadedFile object itself was the original bug.
    image_bytes = uploaded.getvalue()

    with st.spinner(f"Asking {model}..."):
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": question or "Describe this image.",
                        "images": [image_bytes],
                    }
                ],
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Vision model call failed: {type(exc).__name__}: {exc}")
            return

    st.markdown(response["message"]["content"])
