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
