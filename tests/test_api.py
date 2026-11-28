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
    api._USAGE.clear()
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


def _two_docs(client, monkeypatch, sample_docs, headers=None):
    """Ingest a.pdf and b.pdf; each gets a retriever returning its own passage."""
    monkeypatch.setenv("AURALITH_RERANK_PROVIDER", "none")
    monkeypatch.setattr("auralith.api.load_pdf", lambda path, doc_id: sample_docs)
    report = IndexBuildReport("k", 7, 0.1, False, {})
    monkeypatch.setattr("auralith.api.build_index", lambda *a, **k: (object(), report))
    made = iter("abc")
    monkeypatch.setattr(
        "auralith.api.build_retriever",
        lambda *a, **k: FakeRetriever(
            [sample_docs[0].model_copy(update={"metadata": {
                **sample_docs[0].metadata, "doc_id": next(made)}})]
        ),
    )
    fake = FakeChatModel("both [1][2]")
    monkeypatch.setattr("auralith.rag.get_chat_model", lambda s: fake)
    ids = []
    for name, body in (("a.pdf", PDF), ("b.pdf", PDF + b"2")):
        up = {"file": (name, body, "application/pdf")}
        ids.append(client.post("/ingest", files=up, headers=headers).json()["doc_id"])
    return ids, fake


def test_ask_across_two_docs(client, monkeypatch, sample_docs):
    ids, _ = _two_docs(client, monkeypatch, sample_docs)
    r = client.post("/ask", json={"doc_ids": ids, "question": "layers?"})
    ps = json.loads(r.text.splitlines()[0])["passages"]
    assert sorted(p["filename"] for p in ps) == ["a", "b"]
    assert sorted(p["source_doc_id"] for p in ps) == sorted(ids)


def test_ask_doc_ids_with_foreign_doc_404(client, monkeypatch, sample_docs):
    a, b = _keys(monkeypatch)
    ids, _ = _two_docs(client, monkeypatch, sample_docs, headers=a)
    other = client.post(
        "/ingest", files={"file": ("c.pdf", PDF + b"3", "application/pdf")}, headers=b
    ).json()["doc_id"]
    r = client.post("/ask", json={"doc_ids": [ids[0], other], "question": "q"}, headers=a)
    assert r.status_code == 404


def test_ask_history_limits(client, monkeypatch, sample_docs):
    ids, fake = _two_docs(client, monkeypatch, sample_docs)
    turn = {"question": "q", "answer": "a"}
    r = client.post("/ask", json={"doc_ids": ids, "question": "q", "history": [turn] * 7})
    assert r.status_code == 422
    long = {"question": "q", "answer": "x" * (api.MAX_QUESTION_CHARS + 500)}
    r = client.post("/ask", json={"doc_ids": ids, "question": "and b?", "history": [long]})
    assert r.status_code == 200
    assert len(fake.calls) == 2  # condense + answer
    assert "x" * (api.MAX_QUESTION_CHARS + 1) not in fake.calls[0]


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


def test_request_id_and_access_log(client, monkeypatch, caplog):
    a, _ = _keys(monkeypatch)
    r = client.get("/documents", headers={**a, "X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    bad = client.get("/health", headers={"X-Request-ID": "x\ny"}).headers["X-Request-ID"]
    assert bad != "x\ny" and len(bad) == 16
    with caplog.at_level("INFO", logger="auralith.api"):
        client.get("/documents", headers={**a, "X-Request-ID": "r1"})
    line = json.loads(caplog.records[-1].getMessage())
    assert line["rid"] == "r1" and line["user"] == "alice" and line["status"] == 200


def test_unhandled_error_500_with_request_id(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")

    monkeypatch.setattr("auralith.api.run_graph", boom)
    c = TestClient(api.app, raise_server_exceptions=False)
    r = c.post("/chat", json={"query": "hi"}, headers={"X-Request-ID": "r2"})
    assert r.status_code == 500 and r.json()["request_id"] == "r2"
    assert r.headers["X-Request-ID"] == "r2"


def test_rate_limit_per_user_429(client, monkeypatch):
    a, b = _keys(monkeypatch)
    monkeypatch.setitem(api.LIMITS, "llm", [(2, 60), (0, 86400)])
    monkeypatch.setattr(
        "auralith.api.run_graph", lambda q, s, thread_id=None: {"route": "converse"}
    )
    monkeypatch.setattr("auralith.api.summarise", lambda st: {})
    for _ in range(2):
        assert client.post("/chat", json={"query": "hi"}, headers=a).status_code == 200
    r = client.post("/chat", json={"query": "hi"}, headers=a)
    assert r.status_code == 429 and 1 <= int(r.headers["Retry-After"]) <= 61
    assert client.post("/chat", json={"query": "hi"}, headers=b).status_code == 200


def test_rate_limiter_outage_fails_open(client, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")

    def down(*a, **k):
        raise OSError("db down")

    monkeypatch.setattr("auralith.api.count_usage", down)
    api._rate_limit(Settings.from_env(), "alice", "llm")  # no raise


def test_admin_keys_access(client, monkeypatch):
    a, b = _keys(monkeypatch)
    monkeypatch.setenv("AURALITH_ADMINS", "alice")
    body = {"user": "carol"}
    assert client.post("/admin/keys", json=body).status_code == 401
    assert client.post("/admin/keys", json=body, headers=b).status_code == 403
    assert client.post("/admin/keys", json=body, headers=a).status_code == 501  # no DB
    assert client.post("/admin/keys", json={"user": "Bad Name"}, headers=a).status_code == 422


def test_ingest_provider_error_is_503_retry_after(client, monkeypatch, sample_docs):
    from auralith.providers import ProviderError

    def boom(*a, **k):
        raise ProviderError("Gemini embeddings unavailable after retries (HTTP 429).")

    monkeypatch.setattr("auralith.api.load_pdf", lambda path, doc_id: sample_docs)
    monkeypatch.setattr("auralith.api.build_index", boom)
    r = client.post("/ingest", files={"file": ("a.pdf", PDF, "application/pdf")})
    assert r.status_code == 503 and r.headers["retry-after"] == "60"


@pytest.fixture(autouse=True)
def _guest_off(monkeypatch):
    monkeypatch.delenv("AURALITH_GUEST_ENABLED", raising=False)


def test_client_ip_rightmost_xff_then_fallback():
    from types import SimpleNamespace as NS

    def req(xff):
        h = {"x-forwarded-for": xff} if xff is not None else {}
        return NS(headers=h, client=NS(host="9.9.9.9"))

    assert api._client_ip(req("1.1.1.1, 2.2.2.2 ,  3.3.3.3 ")) == "3.3.3.3"
    assert api._client_ip(req("6.6.6.6")) == "6.6.6.6"
    assert api._client_ip(req(None)) == "9.9.9.9"
    assert api._client_ip(req("")) == "9.9.9.9"


def test_limit_plan_guest_vs_named():
    assert api._limit_plan("alice", "llm") == [("alice", api.LIMITS["llm"])]
    assert api._limit_plan("guest-ab", "ingest") == [
        ("guest-ab", api.GUEST_LIMITS["ingest"]),
        ("*guests", api.GUEST_POOL["ingest"]),
    ]


def test_guest_pool_cap_shared_and_named_unaffected(monkeypatch):
    monkeypatch.setitem(api.GUEST_LIMITS, "llm", [(5, 60)])
    monkeypatch.setitem(api.GUEST_POOL, "llm", [(2, 86400)])
    s = Settings.from_env()
    api._rate_limit(s, "guest-a", "llm")
    api._rate_limit(s, "guest-b", "llm")
    with pytest.raises(api.HTTPException) as e:
        api._rate_limit(s, "guest-c", "llm")
    assert e.value.status_code == 429
    api._rate_limit(s, "alice", "llm")  # named users are outside the pool


def test_guest_per_user_limit_lower(monkeypatch):
    monkeypatch.setitem(api.GUEST_LIMITS, "llm", [(1, 60)])
    monkeypatch.setitem(api.GUEST_POOL, "llm", [(100, 86400)])
    s = Settings.from_env()
    api._rate_limit(s, "guest-a", "llm")
    with pytest.raises(api.HTTPException) as e:
        api._rate_limit(s, "guest-a", "llm")
    assert e.value.status_code == 429
    api._rate_limit(s, "guest-b", "llm")


def test_guest_fails_closed_named_fails_open(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x/y")

    def down(*a, **k):
        raise OSError("db down")

    monkeypatch.setattr("auralith.api.count_usage", down)
    s = Settings.from_env()
    api._rate_limit(s, "alice", "llm")  # no raise
    with pytest.raises(api.HTTPException) as e:
        api._rate_limit(s, "guest-a", "llm")
    assert e.value.status_code == 503 and "Retry-After" in e.value.headers


def test_guest_endpoint_disabled_404_and_no_db_501(client, monkeypatch):
    assert client.post("/guest").status_code == 404
    monkeypatch.setenv("AURALITH_GUEST_ENABLED", "1")
    assert client.post("/guest").status_code == 501


def test_admin_rejects_guest_prefix(client, monkeypatch):
    a, _ = _keys(monkeypatch)
    monkeypatch.setenv("AURALITH_ADMINS", "alice")
    r = client.post("/admin/keys", json={"user": "guest-abc"}, headers=a)
    assert r.status_code == 422
    r = client.post("/admin/keys", json={"user": "sample"}, headers=a)
    assert r.status_code == 501  # allowed, then no DB


def test_sample_disabled_404_and_ask_fallback(client, monkeypatch, sample_docs):
    api.DOCS["abc"] = {
        "retriever": FakeRetriever(sample_docs), "sparse": None,
        "info": {"doc_id": "abc", "owner": "sample", "filename": "m"},
    }
    monkeypatch.setattr("auralith.rag.get_chat_model", lambda s: FakeChatModel("ok [1]"))
    assert client.get("/sample").status_code == 404
    body = {"doc_id": "abc", "question": "q"}
    assert client.post("/ask", json=body).status_code == 404  # no fallback when off
    monkeypatch.setenv("AURALITH_GUEST_ENABLED", "1")
    assert client.get("/sample").json()["doc_id"] == "abc"
    assert client.post("/ask", json=body).status_code == 200


PNG = b"\x89PNG\r\n\x1a\n fake"


def test_image_answers_with_inline_image(client, monkeypatch):
    seen = {}

    class VisionModel:
        def invoke(self, msgs, **_):
            seen["content"] = msgs[0].content
            return FakeResponse("A cat. ")

    def fake_model(settings, model=None):
        seen["model"] = model
        return VisionModel()

    monkeypatch.setattr("auralith.api.get_chat_model", fake_model)
    monkeypatch.setenv("AURALITH_GENERATOR_PROVIDER", "ollama")
    monkeypatch.setenv("AURALITH_PROVIDER", "ollama")
    r = client.post("/image", files={"file": ("c.png", PNG, "image/png")}, data={"question": "What?"})
    assert r.status_code == 200 and r.json()["answer"] == "A cat."
    assert seen["content"][0] == {"type": "text", "text": "What?"}
    assert seen["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert seen["model"] == "llava"  # default Ollama provider: vision model, not the 3B text one

    monkeypatch.setenv("AURALITH_GENERATOR_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    client.post("/image", files={"file": ("c.png", PNG, "image/png")})
    assert seen["model"] is None  # hosted generator reads the image itself


def test_image_rejects_non_image_and_too_large(client, monkeypatch):
    r = client.post("/image", files={"file": ("x.png", b"%PDF-1.4", "image/png")})
    assert r.status_code == 415
    monkeypatch.setattr("auralith.api.MAX_IMAGE_BYTES", 10)
    r = client.post("/image", files={"file": ("x.png", PNG + b"x" * 20, "image/png")})
    assert r.status_code == 413


def test_image_model_failure_is_502(client, monkeypatch):
    class Broken:
        def invoke(self, *_a, **_k):
            raise RuntimeError("quota")

    monkeypatch.setattr("auralith.api.get_chat_model", lambda s, model=None: Broken())
    r = client.post("/image", files={"file": ("c.png", PNG, "image/png")})
    assert r.status_code == 502 and "quota" in r.json()["detail"]


def test_image_counts_against_llm_rate_limit(client, monkeypatch):
    monkeypatch.setattr("auralith.api.get_chat_model", lambda s, model=None: FakeChatModel("ok"))
    monkeypatch.setitem(api.LIMITS, "llm", [(1, 60)])
    up = {"file": ("c.png", PNG, "image/png")}
    assert client.post("/image", files=up).status_code == 200
    assert client.post("/image", files=up).status_code == 429
