"""answer_question retrieval fallback: only retrieval errors fall back."""

from auralith.config import RetrievalConfig, Settings
from auralith import rag


class Down:
    def invoke(self, q, **_):
        raise ConnectionError("db down")


class Ok:
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, q, **_):
        return self.docs


class Model:
    def invoke(self, prompt, **_):
        return type("R", (), {"content": "answer [1]"})()


def test_fallback_sets_degraded(monkeypatch, sample_docs):
    monkeypatch.setattr(rag, "get_chat_model", lambda s: Model())
    r = rag.answer_question(
        "q", Down(), RetrievalConfig(k=2), Settings(), fallback=Ok(sample_docs)
    )
    assert r.error is None and r.passages
    assert r.meta["degraded"].startswith("ConnectionError")


def test_no_fallback_keeps_error_result(monkeypatch):
    monkeypatch.setattr(rag, "get_chat_model", lambda s: Model())
    r = rag.answer_question("q", Down(), RetrievalConfig(k=2), Settings())
    assert r.error.startswith("ConnectionError") and r.meta == {}
