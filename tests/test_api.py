"""API tests: fake model and fake index, no network, no Ollama."""

from __future__ import annotations

import json

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
    api.DOCS.clear()
    monkeypatch.setattr("auralith.api.check_backend", lambda s: None)
    return TestClient(api.app)


PDF = b"%PDF-1.4 fake"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
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
    r = client.post("/chat", json={"query": "hi"})
    assert r.status_code == 200
    assert r.json()["route"] == "Converse"
    assert r.json()["answer"] == "Hello!"
