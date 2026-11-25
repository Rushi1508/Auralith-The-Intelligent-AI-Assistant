"""Tests for the path eval: request-set integrity, code runner, search cache, scoring."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import auralith.graph.nodes as nodes
from auralith.tools.search import SearchResponse, SearchResult
from eval.compare import compare
from eval.eval_paths import PATHS, cached_search, run_one, run_python, score
from eval.judge import judge_answer
from tests.test_graph import FakeChatModel

EVAL = Path(__file__).resolve().parent.parent / "eval"


def _load(name: str) -> list[dict]:
    return json.loads((EVAL / name).read_text(encoding="utf-8"))["requests"]


def test_request_sets_valid():
    main, held = _load("paths.json"), _load("paths_heldout.json")
    for items, per_path in ((main, 18), (held, 8)):
        assert len({i["id"] for i in items}) == len(items)
        assert Counter(i["path"] for i in items) == {p: per_path for p in PATHS}
        assert all(i["reference"].strip() for i in items)
        for i in items:
            if i["path"] == "SearchWeb":
                assert i["answer_keys"]
            if i["path"] == "WriteCode":
                assert i["language"] and "test" in i
                if i["language"] == "python":
                    assert i["test"]
    assert not {i["id"] for i in main} & {i["id"] for i in held}
    assert not {i["query"] for i in main} & {i["query"] for i in held}
    routes = {
        r["query"]
        for f in ("routes.json", "routes_heldout.json")
        for r in json.loads((EVAL / f).read_text(encoding="utf-8"))["routes"]
    }
    assert not routes & {i["query"] for i in main + held}


def test_run_python_pass_fail_timeout_stdin():
    assert run_python("x = 1", "assert x == 1")[0] is True
    ok, err = run_python("x = 1", "assert x == 2")
    assert ok is False and "AssertionError" in err
    assert run_python("while True: pass", "", timeout=1) == (False, "timeout")
    assert run_python("input()", "")[0] is False  # stdin is closed, fails fast


def test_cached_search_replays_and_skips_errors(tmp_path):
    calls = []

    def live(query, *a, **k):
        calls.append(query)
        if query == "bad":
            return SearchResponse(query=query, error="rate limited")
        return SearchResponse(query, [SearchResult("t", "http://u", "s")], backend="x")

    search = cached_search(tmp_path / "c.json", live=live)
    first = search("good")
    again = search("good")
    assert again.results == first.results and again.backend == "x"
    search("bad")
    search("bad")
    assert calls == ["good", "bad", "bad"]


def test_run_one_each_path(monkeypatch):
    monkeypatch.setattr(
        nodes,
        "search_web",
        lambda q: SearchResponse(q, [SearchResult("Title", "http://u", "Paris is it")]),
    )
    s = run_one(
        {"path": "SearchWeb", "query": "q", "answer_keys": ["paris"]},
        FakeChatModel("Paris"),
    )
    assert s["answer"] == "Paris" and s["search_ok"] and s["results_keyword_score"] == 1.0

    item = {"path": "WriteCode", "query": "q", "language": "python",
            "test": "assert f() == 1"}
    c = run_one(item, FakeChatModel("```python\ndef f():\n    return 1\n```"))
    assert c["code_passes"] is True and c["truncated"] is False
    c = run_one(item, FakeChatModel("```python\ndef f():\n    return 2\n```"))
    assert c["code_passes"] is False
    # No closing fence.
    c = run_one(item, FakeChatModel("```python\ndef f():\n    return 1"))
    assert c["truncated"] is True and c["code_passes"] is False
    sql = run_one({"path": "WriteCode", "query": "q", "language": "sql", "test": None},
                  FakeChatModel("```sql\nSELECT 1;\n```"))
    assert sql["code_passes"] is None

    v = run_one({"path": "Converse", "query": "q"}, FakeChatModel("Hello"))
    assert v["answer"] == "Hello" and v["error"] is None


def test_judge_keeps_indexing_when_not_stripping():
    verdict = '{"required_facts": [], "missing_or_wrong": [], "correct": true}'
    kept, stripped = FakeChatModel(verdict), FakeChatModel(verdict)
    judge_answer(kept, "q", "r", "x = a[0]", strip_citations=False)
    judge_answer(stripped, "q", "r", "x = a[0]")
    assert "x = a[0]" in kept.calls[0]
    assert "x = a[0]" not in stripped.calls[0]


def test_score_per_path():
    rows = [
        {"path": "WriteCode", "judge_score": 1.0, "code_passes": True},
        {"path": "WriteCode", "judge_score": 0.0, "code_passes": False,
         "truncated": True},
        {"path": "WriteCode", "judge_score": 1.0, "code_passes": None},
        {"path": "Converse", "judge_score": 0.0, "judge_parse_failed": True,
         "error": "x"},
    ]
    s = score(rows)
    assert s["WriteCode"]["code_pass_rate"] == 0.5
    assert s["WriteCode"]["code_pass_n"] == 2 and s["WriteCode"]["truncated"] == 1
    assert s["WriteCode"]["judge_accuracy"] == 0.6667
    assert s["Converse"]["judge_parse_failed"] == 1 and s["Converse"]["errors"] == 1
    assert s["overall"]["n"] == 4 and "SearchWeb" not in s


def test_compare_per_path_metrics():
    def run(sw, wc):
        return {"questions": [
            {"id": "a", "path": "SearchWeb", "answerable": True, "judge_score": sw},
            {"id": "b", "path": "WriteCode", "answerable": True, "judge_score": wc,
             "code_passes": bool(wc)},
        ]}

    labels = {r["metric"]: r for r in compare(run(0.0, 0.0), run(1.0, 1.0))}
    assert labels["answer_correctness (SearchWeb)"]["n"] == 1
    assert labels["answer_correctness (WriteCode)"]["delta"] == 1.0
    assert labels["code_passes (WriteCode)"]["delta"] == 1.0
    assert "answer_correctness (Converse)" not in labels
    assert "recall@k" not in labels
