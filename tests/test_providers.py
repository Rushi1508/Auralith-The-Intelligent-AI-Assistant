"""Hosted-generator wiring: the app must not need Ollama when Gemini is the generator."""

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
