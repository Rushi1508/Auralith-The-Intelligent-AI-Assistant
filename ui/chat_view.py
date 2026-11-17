"""Chat view, backed by the LangGraph agent.

The inherited chat_module held routing, prompting, voice capture, file writing
and Streamlit widgets in one 128-line function. This view does presentation
only: it collects a query, calls `run_graph`, and renders the result. The
routing logic it replaced is now unit-tested in tests/test_graph.py, which was
impossible when it lived inline here.

Voice capture is kept, with the reruns bug fixed. In the prototype the
transcript was processed in the same run that produced it and then lost on the
next rerun, so the Submit button could never act on a voice prompt. It now goes
through session state.
"""

from __future__ import annotations

import streamlit as st

from auralith.config import Settings
from auralith.graph.build import run_graph, summarise
from auralith.providers import ProviderError, check_backend


def _transcribe() -> str:
    """Capture one utterance from the microphone.

    Optional: speech_recognition needs PyAudio, which is a separate system-level
    install. A missing microphone stack degrades to a message rather than an
    ImportError traceback on app start.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        st.error(
            "Voice input needs SpeechRecognition and PyAudio: "
            "pip install SpeechRecognition pyaudio"
        )
        return ""

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            st.info("Listening...")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)
    except OSError as exc:
        st.error(f"No usable microphone: {exc}")
        return ""

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.warning("Could not understand the audio.")
    except sr.RequestError as exc:
        st.error(f"Speech service unavailable: {exc}")
    return ""


def start_chat_module() -> None:
    st.title("Auralith")
    st.caption("Chat Mode - routed by a LangGraph state machine")

    settings = Settings.from_env()
    try:
        check_backend(settings)
    except ProviderError as exc:
        st.error(str(exc))
        return

    st.session_state.setdefault("chat_prompt", "")

    if st.button("Capture voice"):
        # Written to session state, not a local, so it survives the rerun that
        # the button press itself triggers.
        transcript = _transcribe()
        if transcript:
            st.session_state.chat_prompt = transcript

    prompt = st.text_area(
        "Your request",
        value=st.session_state.chat_prompt,
        placeholder="Ask a question, request code, or ask for something to be looked up.",
        height=110,
    )

    if not st.button("Submit", type="primary"):
        return
    if not prompt.strip():
        st.warning("Enter a prompt first.")
        return

    with st.spinner("Routing..."):
        state = run_graph(prompt, settings)
    summary = summarise(state)

    cols = st.columns(3)
    cols[0].metric("Route", summary["route"] or "—")
    cols[1].metric("Recovered", "yes" if summary["recovered"] else "no")
    cols[2].metric("Retries", summary["retries"])

    if summary["answer"]:
        st.markdown(summary["answer"])
    if summary["error"]:
        st.warning(f"Note: {summary['error']}")

    if state.get("search_results"):
        with st.expander(f"Web sources ({len(state['search_results'])})"):
            for i, r in enumerate(state["search_results"], 1):
                st.markdown(f"**[{i}] [{r['title']}]({r['url']})**")
                st.caption(r["snippet"])

    if state.get("code"):
        st.download_button(
            "Download code",
            data=state["code"],
            file_name=f"auralith_snippet.{state.get('language', 'txt')}",
        )

    # The execution trace. Makes the routing decision inspectable instead of
    # something the user has to infer from the shape of the answer.
    with st.expander("Graph trace"):
        st.code(summary["trace"] or "(empty)")
