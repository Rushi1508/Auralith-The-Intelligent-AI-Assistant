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


def test_passages_shown_in_reading_order_with_relevance_numbers():
    from langchain_core.documents import Document

    from auralith.rag import format_passages

    docs = [Document(page_content=f"text{i}", metadata={"doc_id": "cv", "page": 1, "chunk_index": i})
            for i in (2, 0, 1)]  # relevance order
    text = format_passages(docs)
    assert text.index("[2] (source: cv, page 1, part 1)") < text.index("[3] (source: cv, page 1, part 2)") \
        < text.index("[1] (source: cv, page 1, part 3)")


def test_condense_no_history_makes_no_llm_call(monkeypatch):
    from auralith.config import Settings
    from auralith.rag import condense_question

    def boom(s):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr("auralith.rag.get_chat_model", boom)
    assert condense_question([], "what?", Settings.from_env()) == "what?"


def test_condense_falls_back_when_model_raises(monkeypatch):
    from auralith.config import Settings
    from auralith.rag import condense_question

    class Broken:
        def invoke(self, *_a, **_k):
            raise RuntimeError("down")

    monkeypatch.setattr("auralith.rag.get_chat_model", lambda s: Broken())
    turn = {"question": "q", "answer": "a"}
    assert condense_question([turn], "and then?", Settings.from_env()) == "and then?"


def test_condense_strips_prefix_and_quotes(monkeypatch):
    from auralith.config import Settings
    from auralith.rag import condense_question

    class M:
        def invoke(self, *_a, **_k):
            return type("R", (), {"content": 'Standalone question: "What is b?"'})()

    monkeypatch.setattr("auralith.rag.get_chat_model", lambda s: M())
    turn = {"question": "q", "answer": "a"}
    assert condense_question([turn], "and b?", Settings.from_env()) == "What is b?"
