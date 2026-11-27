"""Hosted-generator wiring: the app must not need Ollama when Gemini is the generator."""

import sys
import types

import numpy as np
import pytest

from auralith.config import Settings
from auralith.providers import ProviderError, check_backend

DEAD_OLLAMA = "http://127.0.0.1:9"  # nothing listens here


def test_gemini_needs_no_ollama_for_the_app():
    s = Settings(generator_provider="gemini", gemini_api_key="k", ollama_base_url=DEAD_OLLAMA)
    check_backend(s)  # no exception: embeddings/reranker are local


def test_gemini_without_key_fails_clearly():
    s = Settings(generator_provider="gemini", gemini_api_key=None, ollama_base_url=DEAD_OLLAMA)
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        check_backend(s)


def test_eval_still_requires_ollama_for_the_judge():
    s = Settings(generator_provider="gemini", gemini_api_key="k", ollama_base_url=DEAD_OLLAMA)
    with pytest.raises(ProviderError, match="Cannot reach Ollama"):
        check_backend(s, need_judge=True)


def test_gemini_defaults_to_a_gemini_model(monkeypatch):
    monkeypatch.setenv("AURALITH_GENERATOR_PROVIDER", "gemini")
    monkeypatch.delenv("AURALITH_GENERATOR_MODEL", raising=False)
    assert Settings.from_env().generator_model.startswith("gemini")


def test_ollama_model_has_timeout():
    from auralith.providers import get_chat_model

    get_chat_model.cache_clear()
    m = get_chat_model(Settings(request_timeout=7))
    assert m.client_kwargs == {"timeout": 7}


def _stub_fastembed(monkeypatch):
    """Fake fastembed so the adapter logic is tested without downloading models."""
    seen = {}

    class TextEmbedding:
        def __init__(self, name, threads=None, **kw):
            seen["emb"] = name

        def embed(self, texts, batch_size=256):
            return (np.array([float(len(t)), 1.0]) for t in texts)

    class TextCrossEncoder:
        def __init__(self, name, threads=None, **kw):
            seen["ce"] = name

        def rerank_pairs(self, pairs, batch_size=8):
            return (float(len(d)) for _, d in pairs)

    fe = types.ModuleType("fastembed")
    fe.TextEmbedding = TextEmbedding
    ce = types.ModuleType("fastembed.rerank.cross_encoder")
    ce.TextCrossEncoder = TextCrossEncoder
    monkeypatch.setitem(sys.modules, "fastembed", fe)
    monkeypatch.setitem(sys.modules, "fastembed.rerank", types.ModuleType("fastembed.rerank"))
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", ce)
    from auralith.providers import get_cross_encoder, get_embeddings

    get_embeddings.cache_clear()
    get_cross_encoder.cache_clear()
    return seen


def test_onnx_embeddings_are_float_lists(monkeypatch):
    from auralith.providers import get_embeddings

    seen = _stub_fastembed(monkeypatch)
    emb = get_embeddings(Settings(ml_backend="onnx"))
    assert seen["emb"] == "sentence-transformers/all-MiniLM-L6-v2"
    assert emb.embed_documents(["ab", "abc"]) == [[2.0, 1.0], [3.0, 1.0]]
    assert emb.embed_query("abcd") == [4.0, 1.0]


def test_onnx_cross_encoder_maps_name_and_reranks(monkeypatch):
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
    from langchain_core.cross_encoders import BaseCrossEncoder
    from langchain_core.documents import Document

    from auralith.providers import get_cross_encoder

    seen = _stub_fastembed(monkeypatch)
    ce = get_cross_encoder(Settings(ml_backend="onnx"))
    assert isinstance(ce, BaseCrossEncoder)
    assert seen["ce"] == "Xenova/ms-marco-MiniLM-L-6-v2"
    docs = [Document(page_content=t) for t in ("a", "ccc", "bb")]
    out = CrossEncoderReranker(model=ce, top_n=2).compress_documents(docs, "q")
    assert [d.page_content for d in out] == ["ccc", "bb"]


def test_ml_backend_env(monkeypatch):
    monkeypatch.delenv("AURALITH_ML_BACKEND", raising=False)
    assert Settings.from_env().ml_backend == "onnx"
    monkeypatch.setenv("AURALITH_ML_BACKEND", "torch")
    assert Settings.from_env().ml_backend == "torch"


# --- Gemini embeddings (no network: httpx.post and time.sleep are faked) ---------

class _Resp:
    def __init__(self, status=200, vectors=None, headers=None):
        self.status_code, self.headers, self.text = status, headers or {}, "x"
        self._v = vectors

    def json(self):
        return {"embeddings": [{"values": v} for v in self._v]}


def _gemini(monkeypatch, handler):
    import httpx

    from auralith import providers

    calls = []

    def post(url, json, headers, timeout):
        calls.append((url, json, headers))
        return handler(len(calls), json)

    monkeypatch.setattr(httpx, "post", post)
    monkeypatch.setattr(providers.time, "sleep", lambda s: calls.append(("sleep", s)))
    return providers._GeminiEmbeddings("gemini-embedding-001", "KEY", 5), calls


def _ok(n, body):
    return _Resp(vectors=[[3.0, 4.0]] * len(body["requests"]))


def test_gemini_request_shape_and_normalised_output(monkeypatch):
    emb, calls = _gemini(monkeypatch, _ok)
    assert emb.embed_query("q") == [[0.6, 0.8]][0]
    url, body, headers = calls[0]
    assert url.endswith("/models/gemini-embedding-001:batchEmbedContents")
    assert headers == {"x-goog-api-key": "KEY"}
    r = body["requests"][0]
    assert r["taskType"] == "RETRIEVAL_QUERY" and r["outputDimensionality"] == 768
    assert r["model"] == "models/gemini-embedding-001"
    emb.embed_documents(["d"])
    assert calls[1][1]["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"


def test_gemini_batches_of_32(monkeypatch):
    emb, calls = _gemini(monkeypatch, _ok)
    assert len(emb.embed_documents([str(i) for i in range(70)])) == 70
    assert [len(c[1]["requests"]) for c in calls] == [32, 32, 6]


def test_gemini_retries_429_then_succeeds(monkeypatch):
    emb, calls = _gemini(
        monkeypatch,
        lambda n, b: _Resp(429, headers={"Retry-After": "7"}) if n == 1 else _ok(n, b),
    )
    assert len(emb.embed_documents(["a"])) == 1
    assert ("sleep", 7.0) in calls


def test_gemini_constant_429_raises_never_partial(monkeypatch):
    emb, _ = _gemini(monkeypatch, lambda n, b: _Resp(429))
    with pytest.raises(ProviderError, match="after retries"):
        emb.embed_documents(["a"])


def test_gemini_4xx_not_retried_and_not_retryable(monkeypatch):
    emb, calls = _gemini(monkeypatch, lambda n, b: _Resp(403))
    with pytest.raises(ProviderError, match="HTTP 403") as ei:
        emb.embed_documents(["a"])
    assert not ei.value.retryable and len(calls) == 1


def test_gemini_query_retries_twice_and_length_mismatch_raises(monkeypatch):
    emb, calls = _gemini(monkeypatch, lambda n, b: _Resp(500))
    with pytest.raises(ProviderError, match="after retries") as ei:
        emb.embed_query("q")
    assert ei.value.retryable and len([c for c in calls if c[0] != "sleep"]) == 2
    emb, _ = _gemini(monkeypatch, lambda n, b: _Resp(vectors=[[1.0, 0.0]]))
    with pytest.raises(ProviderError, match="1 embeddings for 2 inputs"):
        emb.embed_documents(["a", "b"])


def test_unknown_rerank_provider_rejected(monkeypatch):
    from auralith.providers import get_cross_encoder

    monkeypatch.setenv("AURALITH_RERANK_PROVIDER", "gemini")
    with pytest.raises(ValueError, match="AURALITH_RERANK_PROVIDER"):
        Settings.from_env()
    with pytest.raises(ProviderError, match="Unknown rerank_provider"):
        get_cross_encoder(Settings(rerank_provider="gemini"))  # type: ignore[arg-type]


def test_gemini_settings_and_missing_key(monkeypatch):
    from auralith.providers import get_embeddings

    monkeypatch.setenv("AURALITH_EMBEDDING_PROVIDER", "gemini")
    monkeypatch.delenv("AURALITH_EMBEDDING_MODEL", raising=False)
    s = Settings.from_env()
    assert s.embedding_model == "gemini-embedding-001"
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        get_embeddings(Settings(embedding_provider="gemini"))
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        check_backend(Settings(embedding_provider="gemini"))


def test_embed_tag_keeps_local_names_and_separates_gemini():
    from auralith.config import PRODUCTION_CONFIG
    from auralith.index import collection_name, embed_tag

    assert embed_tag(Settings()) == ""
    assert collection_name("x", PRODUCTION_CONFIG, Settings()) == "x__recursive_cs500_co100"
    g = Settings(embedding_provider="gemini", embedding_model="gemini-embedding-001")
    assert collection_name("x", PRODUCTION_CONFIG, g) == (
        "x__recursive_cs500_co100__gemini-embedding-001"
    )


def test_rerank_none_keeps_order():
    from auralith.providers import get_cross_encoder

    ce = get_cross_encoder(Settings(rerank_provider="none"))
    scores = ce.score([("q", "a"), ("q", "b"), ("q", "c")])
    assert scores == sorted(scores, reverse=True)


def test_rerank_none_runs_and_labels_as_no_rerank():
    """Render path: no reranker in the chain, k fetched directly, and labels say so."""
    from langchain_core.documents import Document

    from auralith.config import PRODUCTION_CONFIG, effective_config
    from auralith.retrievers import build_retriever

    class FakeStore:
        def as_retriever(self, search_kwargs):
            self.k = search_kwargs["k"]
            return "dense"

    s = Settings(rerank_provider="none")
    cfg = effective_config(PRODUCTION_CONFIG, s)
    assert not cfg.use_rerank and "rr" not in cfg.slug and "rerank" not in cfg.describe()
    assert effective_config(PRODUCTION_CONFIG, Settings()) is PRODUCTION_CONFIG

    store = FakeStore()
    from unittest.mock import patch

    with patch("auralith.retrievers.EnsembleRetriever", create=True), \
         patch("langchain_classic.retrievers.EnsembleRetriever") as ens:
        r = build_retriever(store, cfg, s, bm25_chunks=[Document(page_content="x")])
    assert store.k == PRODUCTION_CONFIG.k
    assert r is ens.return_value  # ensemble on top, no ContextualCompressionRetriever
