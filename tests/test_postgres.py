"""Postgres-backed paths (pgvector + PostgresSaver). Needs DATABASE_URL; skipped otherwise."""

from __future__ import annotations

import os
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from auralith.config import Settings
from auralith.index import _pg_engine, build_index

pytestmark = pytest.mark.skipif(not os.getenv("DATABASE_URL"), reason="DATABASE_URL not set")


@pytest.fixture
def settings(monkeypatch) -> Settings:
    monkeypatch.setattr(
        "auralith.index.get_embeddings", lambda s: DeterministicFakeEmbedding(size=16)
    )
    return Settings.from_env()


@pytest.fixture
def root(tmp_path, settings, base_config):
    """Fresh per-test root; drops every collection it creates."""
    created: list[Path] = []

    def make() -> Path:
        r = tmp_path / uuid.uuid4().hex
        created.append(r)
        return r

    yield make
    from langchain_postgres import PGVector

    for r in created:
        PGVector(
            embeddings=DeterministicFakeEmbedding(size=16),
            collection_name=f"{r.name}__{base_config.index_key}",
            connection=_pg_engine(settings.database_url),
            use_jsonb=True,
        ).delete_collection()


def test_build_then_cache_hit(settings, root, base_config, sample_docs):
    r = root()
    store, report = build_index(sample_docs, base_config, settings, root=r)
    assert not report.from_cache and report.n_chunks > 0
    docs = store.as_retriever(search_kwargs={"k": 2}).invoke("encoder layers")
    assert docs and all("doc_id" in d.metadata for d in docs)

    _, again = build_index(sample_docs, base_config, settings, root=r)
    assert again.from_cache and again.n_chunks == report.n_chunks


def test_two_roots_do_not_collide(settings, root, base_config, sample_docs):
    for _ in range(2):
        _, report = build_index(sample_docs, base_config, settings, root=root())
        assert not report.from_cache


def test_thread_state_persists_in_postgres(monkeypatch):
    from langgraph.checkpoint.postgres import PostgresSaver

    from auralith.graph.build import run_graph
    from tests.test_graph import FakeChatModel

    router = '{{"action": "{}", "parameters": "p"}}'
    fake = FakeChatModel(
        router.format("WriteCode"), "```python\nprint(1)\n```",
        router.format("Converse"), "Prints 1.",
    )
    monkeypatch.setattr("auralith.graph.nodes.get_chat_model", lambda s: fake)
    settings = replace(Settings.from_env(), database_url=os.environ["DATABASE_URL"])
    tid = uuid.uuid4().hex

    run_graph("write a print", settings, thread_id=tid)
    t2 = run_graph("what does it do", settings, thread_id=tid)
    assert t2["code"] is None and len(t2["messages"]) == 4

    with PostgresSaver.from_conn_string(settings.database_url) as saver:
        assert saver.get_tuple({"configurable": {"thread_id": tid}}) is not None


@pytest.fixture
def pg_url() -> str:
    return os.environ["DATABASE_URL"]


def test_registry_owner_scoped(pg_url):
    from sqlalchemy import text

    from auralith.registry import ensure_schema, get_doc, save_doc

    doc = uuid.uuid4().hex[:12]
    info = {"doc_id": doc, "filename": "f", "owner": "alice"}
    try:
        save_doc(pg_url, "alice", info)
        save_doc(pg_url, "alice", info)  # idempotent
        assert get_doc(pg_url, "alice", doc) == info
        assert get_doc(pg_url, "bob", doc) is None
    finally:
        ensure_schema(pg_url)
        with _pg_engine(pg_url).begin() as c:
            c.execute(text("DELETE FROM auralith_docs WHERE doc_id = :d"), {"d": doc})


def test_load_pg_chunks_matches_build(settings, root, base_config, sample_docs):
    from auralith.index import collection_name, load_pg_chunks

    r = root()
    _, report = build_index(sample_docs, base_config, settings, root=r)
    chunks = load_pg_chunks(collection_name(r.name, base_config), settings)
    assert len(chunks) == report.n_chunks
    assert all(c.page_content and "chunk_id" in c.metadata for c in chunks)


def _api_client(monkeypatch, sample_docs, model):
    from fastapi.testclient import TestClient

    from auralith import api

    monkeypatch.setattr(
        "auralith.index.get_embeddings", lambda s: DeterministicFakeEmbedding(size=16)
    )
    monkeypatch.setattr("auralith.api.check_backend", lambda s: None)
    monkeypatch.setattr("auralith.api.load_pdf", lambda path, doc_id: sample_docs)
    monkeypatch.setattr(
        "auralith.api.build_retriever", lambda store, *a, **k: store.as_retriever()
    )
    monkeypatch.setattr("auralith.rag.get_chat_model", lambda s: model)
    api._CACHE.clear()
    api.DOCS.clear()
    return TestClient(api.app)


def test_ask_survives_restart(monkeypatch, sample_docs, pg_url):
    import json

    from sqlalchemy import text

    from auralith import api
    from auralith.config import PRODUCTION_CONFIG
    from auralith.registry import hash_key
    from tests.test_api import FakeChatModel

    monkeypatch.setenv(
        "AURALITH_API_KEYS", f"alice:{hash_key('ka')},bob:{hash_key('kb')}"
    )
    client = _api_client(monkeypatch, sample_docs, FakeChatModel("ok [1]"))
    a, b = {"X-API-Key": "ka"}, {"X-API-Key": "kb"}
    pdf = b"%PDF-1.4 " + uuid.uuid4().hex.encode()  # unique bytes: fresh doc_id
    doc_id = client.post(
        "/ingest", files={"file": ("a.pdf", pdf, "application/pdf")}, headers=a
    ).json()["doc_id"]
    try:
        api._CACHE.clear()
        api.DOCS.clear()  # simulate another worker / a restart
        body = {"doc_id": doc_id, "question": "encoder layers"}
        r = client.post("/ask", json=body, headers=a)
        events = [json.loads(line) for line in r.text.splitlines()]
        assert events[0]["passages"] and events[1]["degraded"] is False
        assert client.post("/ask", json=body, headers=b).status_code == 404
    finally:
        from langchain_postgres import PGVector

        s = Settings.from_env()
        PGVector(
            embeddings=DeterministicFakeEmbedding(size=16),
            collection_name=f"{doc_id}__{PRODUCTION_CONFIG.index_key}",
            connection=_pg_engine(s.database_url),
            use_jsonb=True,
        ).delete_collection()
        with _pg_engine(s.database_url).begin() as c:
            c.execute(text("DELETE FROM auralith_docs WHERE doc_id = :d"), {"d": doc_id})


def test_thread_isolation_postgres(monkeypatch, sample_docs, pg_url):
    from langgraph.checkpoint.postgres import PostgresSaver

    from auralith.registry import hash_key
    from tests.test_graph import FakeChatModel

    monkeypatch.setenv("AURALITH_API_KEYS", f"alice:{hash_key('ka')}")
    fake = FakeChatModel('{"action": "Converse", "parameters": "p"}', "Hello!")
    monkeypatch.setattr("auralith.graph.nodes.get_chat_model", lambda s: fake)
    client = _api_client(monkeypatch, sample_docs, fake)
    t = uuid.uuid4().hex
    r = client.post(
        "/chat", json={"query": "hi", "thread_id": t}, headers={"X-API-Key": "ka"}
    )
    assert r.status_code == 200 and r.json()["thread_id"] == t
    with PostgresSaver.from_conn_string(pg_url) as saver:
        get = lambda tid: saver.get_tuple({"configurable": {"thread_id": tid}})  # noqa: E731
        assert get(f"alice:{t}") is not None
        assert get(f"bob:{t}") is None and get(t) is None


def test_concurrent_ingest_same_doc(monkeypatch, sample_docs, pg_url):
    import threading

    from sqlalchemy import text

    from auralith.config import PRODUCTION_CONFIG

    client = _api_client(monkeypatch, sample_docs, None)
    pdf = b"%PDF-1.4 " + uuid.uuid4().hex.encode()
    out = []

    def go():
        r = client.post("/ingest", files={"file": ("a.pdf", pdf, "application/pdf")})
        out.append((r.status_code, r.json().get("doc_id")))

    ts = [threading.Thread(target=go) for _ in range(3)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    doc_id = out[0][1]
    try:
        assert [s for s, _ in out] == [200] * 3 and {d for _, d in out} == {doc_id}
    finally:
        from langchain_postgres import PGVector

        s = Settings.from_env()
        PGVector(
            embeddings=DeterministicFakeEmbedding(size=16),
            collection_name=f"{doc_id}__{PRODUCTION_CONFIG.index_key}",
            connection=_pg_engine(s.database_url),
            use_jsonb=True,
        ).delete_collection()
        with _pg_engine(s.database_url).begin() as c:
            c.execute(text("DELETE FROM auralith_docs WHERE doc_id = :d"), {"d": doc_id})


def test_delete_doc_drops_vectors_and_stale_cache(monkeypatch, sample_docs, pg_url):
    from auralith import api
    from auralith.config import PRODUCTION_CONFIG
    from auralith.index import collection_name, load_pg_chunks
    from tests.test_api import FakeChatModel

    monkeypatch.delenv("AURALITH_API_KEYS", raising=False)
    client = _api_client(monkeypatch, sample_docs, FakeChatModel("ok [1]"))
    pdf = b"%PDF-1.4 " + uuid.uuid4().hex.encode()
    doc_id = client.post(
        "/ingest", files={"file": ("a.pdf", pdf, "application/pdf")}
    ).json()["doc_id"]
    assert doc_id in [d["doc_id"] for d in client.get("/documents").json()]
    stale = dict(api._CACHE)  # another worker's cache still holding the doc
    assert client.delete(f"/documents/{doc_id}").status_code == 204
    api._CACHE.update(stale)
    assert client.post("/ask", json={"doc_id": doc_id, "question": "q"}).status_code == 404
    coll = collection_name(doc_id, PRODUCTION_CONFIG)
    assert load_pg_chunks(coll, Settings.from_env()) == []
    assert doc_id not in [d["doc_id"] for d in client.get("/documents").json()]
