"""Tests for the router eval: label set integrity, recovery mirroring, scoring."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from eval.eval_router import ACTIONS, route_one, score
from tests.test_graph import FakeChatModel

ROUTES = Path(__file__).resolve().parent.parent / "eval" / "routes.json"


def test_label_set_is_valid_and_balanced():
    items = json.loads(ROUTES.read_text(encoding="utf-8"))["routes"]
    assert len({i["id"] for i in items}) == len(items)
    assert set(Counter(i["label"] for i in items).values()) == {20}
    assert {i["label"] for i in items} == set(ACTIONS)
    assert all(i["kind"] in {"clear", "boundary"} for i in items)
    assert all("note" in i for i in items if i["kind"] == "boundary")


def test_route_one_retries_then_falls_back_to_converse():
    llm = FakeChatModel("garbage")
    result = route_one("anything", llm)
    assert result["first"] == "Recover"
    assert result["final"] == "Converse"
    assert len(llm.calls) == 3  # first attempt + MAX_RETRIES


def test_route_one_recovers_on_retry():
    llm = FakeChatModel("garbage", '{"action": "WriteCode", "parameters": "x"}')
    result = route_one("write code", llm)
    assert (result["first"], result["final"], result["retries"]) == ("Recover", "WriteCode", 1)


def test_score_accuracy_and_confusion():
    rows = [
        {"label": "SearchWeb", "final": "SearchWeb", "first": "SearchWeb", "kind": "clear"},
        {"label": "Converse", "final": "WriteCode", "first": "WriteCode", "kind": "boundary"},
        {"label": "WriteCode", "final": "WriteCode", "first": "Recover", "kind": "clear"},
        {"label": "Converse", "final": "Converse", "first": "Converse", "kind": "boundary"},
    ]
    s = score(rows)
    assert s["accuracy"] == 0.75
    assert s["by_kind"] == {"boundary": {"n": 2, "accuracy": 0.5}, "clear": {"n": 2, "accuracy": 1.0}}
    assert s["confusion"]["Converse"]["WriteCode"] == 1
    assert s["first_attempt_unparseable"] == 1
