"""API tests: fake model and fake index, no network, no Ollama."""

from __future__ import annotations

import json
import threading

import pytest
from fastapi.testclient import TestClient

from auralith import api
from auralith.config import Settings
from auralith.index import IndexBuildReport


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeChatModel:
    """Returns scripted responses in order; repeats the last one forever."""

    def __init__(self, *responses: str) -> None:
        self.responses = list(responses) or [""]
        self.calls: list[str] = []

    def invoke(self, prompt, **_: object) -> FakeResponse:
        self.calls.append(str(prompt))
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return FakeResponse(self.responses[idx])


class FakeRetriever:
    def __init__(self, docs) -> None:
        self.docs = docs

    def invoke(self, q, **_: object):
        return self.docs


@pytest.fixture(autouse=True)
def client(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("AURALITH_API_KEYS", raising=False)
    api.DOCS.clear()
    api._CACHE.clear()
    api.DEGRADED.clear()
    monkeypatch.setattr("auralith.api._probe_db", lambda url: None)
    monkeypatch.setattr("auralith.api.check_backend", lambda s: None)
    return TestClient(api.app)


PDF = b"%PDF-1.4 fake"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["checks"] == {}
    assert r.json()["generator_model"] == Settings.from_env().generator_model


def test_ingest_rejects_non_pdf(client):
    r = client.post("/ingest", files={"file": ("notes.txt", b"hi", "text/plain")})
    assert r.status_code == 415
    r = client.post("/ingest", files={"file": ("x.pdf", b"not a pdf", "application/pdf")})
    assert r.status_code == 415


def test_ingest_rejects_too_large(client, monkeypatch):
    monkeypatch.setattr("auralith.api.MAX_UPLOAD_BYTES", 10)
    r = client.post(
        "/ingest", files={"file": ("x.pdf", b"%PDF-" + b"x" * 20, "application/pdf")}
    )
    assert r.status_code == 413


def test_ingest_rejects_pdf_without_text(client, monkeypatch, sample_docs):
    blank = [d.model_copy(update={"page_content": "  "}) for d in sample_docs]
    monkeypatch.setattr("auralith.api.load_pdf", lambda path, doc_id: blank)
    r = client.post("/ingest", files={"file": ("scan.pdf", PDF, "application/pdf")})
    assert r.status_code == 422
    assert "no extractable text" in r.json()["detail"]
    assert api.DOCS == {}


def test_ingest_then_ask_happy_path(client, monkeypatch, sample_docs):
    loads = []
    monkeypatch.setattr(
        "auralith.api.load_pdf", lambda path, doc_id: loads.append(doc_id) or sample_docs
    )
    report = IndexBuildReport("k", 7, 0.1, False, {})
    monkeypatch.setattr("auralith.api.build_index", lambda *a, **k: (object(), report))
    monkeypatch.setattr(
        "auralith.api.build_retriever", lambda *a, **k: FakeRetriever(sample_docs)
    )
    monkeypatch.setattr(
        "auralith.rag.get_chat_model",
        lambda s: FakeChatModel("The encoder has 6 layers [1]."),
    )

    up = {"file": ("../../evil.pdf", PDF, "application/pdf")}
    info = client.post("/ingest", files=up).json()
    assert info["filename"] == "evil" and info["n_chunks"] == 7

    r = client.post("/ask", json={"doc_id": info["doc_id"], "question": "layers?"})
    events = [json.loads(line) for line in r.text.splitlines()]
    assert [e["type"] for e in events] == ["passages", "answer", "done"]
    assert events[0]["passages"][0]["page"] == 3
    assert events[1]["answer"] == "The encoder has 6 layers [1]."
    assert events[1]["abstained"] is False

    again = client.post("/ingest", files=up).json()
    assert again["doc_id"] == info["doc_id"] and len(loads) == 1


def test_ask_unknown_doc_404(client):
    r = client.post("/ask", json={"doc_id": "nope", "question": "q"})
    assert r.status_code == 404


def test_chat_with_fake_graph_model(client, monkeypatch):
    monkeypatch.setattr(
        "auralith.graph.nodes.get_chat_model",
        lambda s: FakeChatModel('{"action": "Converse", "parameters": "hi"}', "Hello!"),
    )
    r = client.post("/chat", json={"query": "hi", "thread_id": "t1"})
    assert r.status_code == 200
    assert r.json()["route"] == "Converse"
    assert r.json()["answer"] == "Hello!"
    assert r.json()["thread_id"] == "t1"


class DownRetriever:
    def invoke(self, q, **_: object):
        raise ConnectionError("db down")


class BrokenModel:
    def invoke(self, prompt, **_: object):
        raise TimeoutError("llm slow")


def _ingest(client, monkeypatch, sample_docs, dense):
    monkeypatch.setattr("auralith.api.load_pdf", lambda path, doc_id: sample_docs)
    report = IndexBuildReport("k", 7, 0.1, False, {})
    monkeypatch.setattr("auralith.api.build_index", lambda *a, **k: (object(), report))
    monkeypatch.setattr("auralith.api.build_retriever", lambda *a, **k: dense)
    up = {"file": ("a.pdf", PDF, "application/pdf")}
    return client.post("/ingest", files=up).json()["doc_id"]


def _ask(client, doc_id):
    r = client.post("/ask", json={"doc_id": doc_id, "question": "encoder layers"})
    return [json.loads(line) for line in r.text.splitlines()]


def test_health_degraded_when_db_down(client, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")
    monkeypatch.setattr("auralith.api._probe_db", lambda url: "OperationalError: x")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"
    assert "database" in r.json()["checks"]


def test_ask_falls_back_to_bm25(client, monkeypatch, sample_docs):
    monkeypatch.setattr("auralith.rag.get_chat_model", lambda s: FakeChatModel("ok [1]"))
    doc_id = _ingest(client, monkeypatch, sample_docs, DownRetriever())
    events = _ask(client, doc_id)
    assert events[0]["passages"]
    assert events[1]["degraded"] is True and events[1]["error"] is None
    h = client.get("/health").json()
    assert h["status"] == "degraded" and "retrieval" in h["checks"]

    api.DOCS[doc_id]["retriever"] = FakeRetriever(sample_docs)
    assert _ask(client, doc_id)[1]["degraded"] is False
    assert client.get("/health").json()["checks"] == {}


def test_ask_llm_error_does_not_fall_back(client, monkeypatch, sample_docs):
    monkeypatch.setattr("auralith.rag.get_chat_model", lambda s: BrokenModel())
    doc_id = _ingest(client, monkeypatch, sample_docs, FakeRetriever(sample_docs))

    class Spy(FakeRetriever):
        called = False

        def invoke(self, q, **_: object):
            Spy.called = True
            return self.docs

    api.DOCS[doc_id]["sparse"] = Spy(sample_docs)
    answer = _ask(client, doc_id)[1]
    assert answer["error"] and answer["degraded"] is False
    assert Spy.called is False


def test_question_and_query_too_long_422(client, monkeypatch, sample_docs):
    doc_id = _ingest(client, monkeypatch, sample_docs, FakeRetriever(sample_docs))
    long = "x" * (api.MAX_QUESTION_CHARS + 1)
    r = client.post("/ask", json={"doc_id": doc_id, "question": long})
    assert r.status_code == 422
    assert client.post("/chat", json={"query": long}).status_code == 422
    r = client.post("/chat", json={"query": "hi", "thread_id": "t" * 129})
    assert r.status_code == 422


def test_busy_returns_503(client, monkeypatch):
    sem = threading.BoundedSemaphore(1)
    sem.acquire()
    monkeypatch.setattr("auralith.api._SLOTS", sem)
    r = client.post("/chat", json={"query": "hi"})
    assert r.status_code == 503 and r.headers["Retry-After"] == "5"


def _keys(monkeypatch):
    from auralith.registry import hash_key

    monkeypatch.setenv(
        "AURALITH_API_KEYS", f"alice:{hash_key('ka')},bob:{hash_key('kb')}"
    )
    return {"X-API-Key": "ka"}, {"X-API-Key": "kb"}


def test_auth_required_except_health(client, monkeypatch):
    _keys(monkeypatch)
    pdf = {"file": ("a.pdf", PDF, "application/pdf")}
    for hdrs in ({}, {"X-API-Key": "wrong"}):
        assert client.post("/ingest", files=pdf, headers=hdrs).status_code == 401
        body = {"doc_id": "x", "question": "q"}
        assert client.post("/ask", json=body, headers=hdrs).status_code == 401
        assert client.post("/chat", json={"query": "hi"}, headers=hdrs).status_code == 401
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["auth"] is True


def test_other_users_doc_404(client, monkeypatch, sample_docs):
    a, b = _keys(monkeypatch)
    monkeypatch.setattr("auralith.api.load_pdf", lambda path, doc_id: sample_docs)
    report = IndexBuildReport("k", 7, 0.1, False, {})
    monkeypatch.setattr("auralith.api.build_index", lambda *a, **k: (object(), report))
    monkeypatch.setattr(
        "auralith.api.build_retriever", lambda *a, **k: FakeRetriever(sample_docs)
    )
    monkeypatch.setattr("auralith.rag.get_chat_model", lambda s: FakeChatModel("ok [1]"))
    up = {"file": ("a.pdf", PDF, "application/pdf")}
    id_a = client.post("/ingest", files=up, headers=a).json()["doc_id"]
    id_b = client.post("/ingest", files=up, headers=b).json()["doc_id"]
    assert id_a != id_b
    body = {"doc_id": id_a, "question": "layers?"}
    assert client.post("/ask", json=body, headers=a).status_code == 200
    assert client.post("/ask", json=body, headers=b).status_code == 404


def test_chat_thread_namespaced(client, monkeypatch):
    a, _ = _keys(monkeypatch)
    seen = []

    def spy(query, settings, thread_id=None, **_):
        seen.append(thread_id)
        return {"answer": "x", "route": "Converse"}

    monkeypatch.setattr("auralith.api.run_graph", spy)
    r = client.post("/chat", json={"query": "hi", "thread_id": "t1"}, headers=a)
    assert r.status_code == 200 and r.json()["thread_id"] == "t1"
    assert seen == ["alice:t1"]


def test_load_keys_rejects_malformed(monkeypatch):
    from auralith.registry import load_keys

    for bad in ("alice", "Alice:" + "a" * 64, "alice:abc", "a b:" + "a" * 64):
        monkeypatch.setenv("AURALITH_API_KEYS", bad)
        with pytest.raises(ValueError):
            load_keys()


def test_malformed_keys_fail_closed(client, monkeypatch):
    monkeypatch.setenv("AURALITH_API_KEYS", "garbage")
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["checks"]["auth"]
    assert r.json()["status"] == "degraded"
    pdf = {"file": ("a.pdf", PDF, "application/pdf")}
    assert client.post("/ingest", files=pdf).status_code == 503
    assert client.post("/ask", json={"doc_id": "x", "question": "q"}).status_code == 503
    assert client.post("/chat", json={"query": "hi"}).status_code == 503


def test_list_and_delete_own_docs_only(client, monkeypatch, sample_docs):
    a, b = _keys(monkeypatch)
    monkeypatch.setattr("auralith.api.load_pdf", lambda path, doc_id: sample_docs)
    report = IndexBuildReport("k", 7, 0.1, False, {})
    monkeypatch.setattr("auralith.api.build_index", lambda *a, **k: (object(), report))
    monkeypatch.setattr(
        "auralith.api.build_retriever", lambda *a, **k: FakeRetriever(sample_docs)
    )
    up = {"file": ("a.pdf", PDF, "application/pdf")}
    id_a = client.post("/ingest", files=up, headers=a).json()["doc_id"]
    assert [d["doc_id"] for d in client.get("/documents", headers=a).json()] == [id_a]
    assert client.get("/documents", headers=b).json() == []
    assert client.delete(f"/documents/{id_a}", headers=b).status_code == 404
    assert client.delete(f"/documents/{id_a}", headers=a).status_code == 204
    assert client.get("/documents", headers=a).json() == []
    body = {"doc_id": id_a, "question": "q"}
    assert client.post("/ask", json=body, headers=a).status_code == 404
    assert client.delete(f"/documents/{id_a}", headers=a).status_code == 404
    assert client.get("/documents").status_code == 401
