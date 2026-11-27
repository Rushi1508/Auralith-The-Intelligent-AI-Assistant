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


def test_passages_carry_reading_order():
    from auralith.config import RetrievalConfig
    from auralith.ingest import chunk_documents
    from langchain_core.documents import Document

    from auralith.rag import format_passages

    page = Document(page_content="\n\n".join(f"Job {i}. " + "word " * 300 for i in range(4)),
                    metadata={"doc_id": "cv", "page": 1})
    chunks = chunk_documents([page], RetrievalConfig())
    assert [c.metadata["chunk_index"] for c in chunks] == list(range(len(chunks)))
    text = format_passages([chunks[-1], chunks[0]])  # relevance order != reading order
    assert f"page 1, part {len(chunks)})" in text and "page 1, part 1)" in text
    assert "(source: x, page 2)" in format_passages([Document(page_content="a", metadata={"doc_id": "x", "page": 2})])
