"""ddgs retry behaviour; DDGS and sleep are faked."""

import sys
import types

from auralith.tools import search


def _fake_ddgs(monkeypatch, outcomes):
    calls = []

    class FakeDDGS:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def text(self, query, max_results):
            calls.append(query)
            out = outcomes[len(calls) - 1]
            if isinstance(out, Exception):
                raise out
            return out

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=FakeDDGS))
    monkeypatch.setattr(search.time, "sleep", lambda s: None)
    return calls


def test_ddgs_retries_once(monkeypatch):
    calls = _fake_ddgs(monkeypatch, [RuntimeError("rate"), [{"title": "t", "href": "u"}]])
    r = search._search_ddgs("q", 5)
    assert len(calls) == 2 and r.ok


def test_ddgs_gives_up_after_two(monkeypatch):
    calls = _fake_ddgs(monkeypatch, [RuntimeError("a"), RuntimeError("b"), []])
    r = search._search_ddgs("q", 5)
    assert len(calls) == 2 and r.error == "RuntimeError: b"
