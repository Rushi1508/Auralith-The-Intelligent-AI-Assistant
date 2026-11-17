"""Tests for graph state, nodes and transitions.

All of these run against a scripted fake model. That is the whole argument for
restructuring the router as a graph: the interesting behaviour is what happens
when the model returns something *wrong*, and with a real model you cannot
reliably produce a malformed response on demand. The prototype's equivalent
logic could not be tested at all without a live API key and a lucky failure.
"""

from __future__ import annotations

from typing import get_type_hints

import pytest

from auralith.graph.nodes import (
    MAX_RETRIES,
    code_node,
    converse_node,
    recovery_decision,
    recovery_node,
    route_decision,
    route_node,
    search_node,
)
from auralith.graph.state import (
    AgentState,
    Route,
    _append,
    _keep_last,
    initial_state,
)


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


class ExplodingChatModel:
    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("backend unreachable")

    def invoke(self, prompt, **_: object):
        raise self.exc


# --------------------------------------------------------------------------
# Route parsing
# --------------------------------------------------------------------------

class TestRouteParsing:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("SearchWeb", Route.SEARCH_WEB),
            ("searchweb", Route.SEARCH_WEB),
            ("SEARCH_WEB", Route.SEARCH_WEB),
            ("search web", Route.SEARCH_WEB),
            ("WriteCode", Route.WRITE_CODE),
            ("write_code", Route.WRITE_CODE),
            ("Converse", Route.CONVERSE),
        ],
    )
    def test_accepts_casing_and_separator_variants(self, raw, expected):
        """Small models are inconsistent about casing and underscores.

        Being strict here would send perfectly good classifications down the
        recovery path.
        """
        assert Route.parse(raw) is expected

    @pytest.mark.parametrize("raw", [None, "", "nonsense", "12345"])
    def test_returns_none_for_unroutable(self, raw):
        assert Route.parse(raw) is None

    def test_finds_action_embedded_in_prose(self):
        assert Route.parse("I think the action is WriteCode here") is Route.WRITE_CODE


# --------------------------------------------------------------------------
# Router node
# --------------------------------------------------------------------------

class TestRouteNode:
    def test_parses_clean_json(self):
        llm = FakeChatModel('{"action": "SearchWeb", "parameters": "weather in Paris"}')
        out = route_node(initial_state("what is the weather in paris"), llm)
        assert out["route"] == Route.SEARCH_WEB.value
        assert out["parameters"] == "weather in Paris"
        assert out["error"] is None

    def test_parses_json_inside_a_fence(self):
        llm = FakeChatModel('```json\n{"action": "WriteCode", "parameters": "a quicksort"}\n```')
        out = route_node(initial_state("write quicksort"), llm)
        assert out["route"] == Route.WRITE_CODE.value

    def test_parses_json_surrounded_by_prose(self):
        llm = FakeChatModel('Sure! {"action": "Converse", "parameters": "hi"} Hope that helps.')
        out = route_node(initial_state("hi"), llm)
        assert out["route"] == Route.CONVERSE.value

    def test_malformed_json_routes_to_recover_instead_of_raising(self):
        """The exact failure that killed a request in the prototype.

        `output_parser.parse(output)` raised a JSONDecodeError straight out of
        the Streamlit handler. Here it becomes a state transition.
        """
        llm = FakeChatModel("I'm not sure what you want, could you clarify?")
        out = route_node(initial_state("???"), llm)
        assert out["route"] == Route.RECOVER.value
        assert out["error"]

    def test_backend_failure_routes_to_recover(self):
        out = route_node(initial_state("anything"), ExplodingChatModel())
        assert out["route"] == Route.RECOVER.value
        assert "Router call failed" in out["error"]

    def test_missing_parameters_falls_back_to_the_raw_query(self):
        llm = FakeChatModel('{"action": "Converse"}')
        out = route_node(initial_state("explain recursion"), llm)
        assert out["parameters"] == "explain recursion"


# --------------------------------------------------------------------------
# Terminal nodes
# --------------------------------------------------------------------------

class TestCodeNode:
    def test_extracts_fenced_code_and_language(self):
        llm = FakeChatModel('```python\nprint("hi")\n```')
        out = code_node({"query": "q", "parameters": "print hi"}, llm)
        assert out["code"] == 'print("hi")'
        assert out["language"] == "python"

    def test_unfenced_output_is_kept_whole(self):
        """The prototype popped first/last lines when they began with a backtick.

        With no closing fence that silently deleted a real line of code. Here an
        unfenced response is preserved intact.
        """
        llm = FakeChatModel('print("hi")\nprint("bye")')
        out = code_node({"query": "q"}, llm)
        assert 'print("bye")' in out["code"]

    def test_backend_failure_is_captured_not_raised(self):
        out = code_node({"query": "q"}, ExplodingChatModel())
        assert out["error"]
        assert "code:error" in out["trace"]


class TestConverseNode:
    def test_returns_answer(self):
        out = converse_node({"query": "q"}, FakeChatModel("Here are the steps."))
        assert out["answer"] == "Here are the steps."
        assert out["error"] is None

    def test_backend_failure_is_captured(self):
        out = converse_node({"query": "q"}, ExplodingChatModel())
        assert out["error"]
        assert out.get("answer") is None


class TestSearchNode:
    def test_failed_search_degrades_with_a_message(self, monkeypatch):
        from auralith.tools import search as search_mod
        from auralith.tools.search import SearchResponse

        monkeypatch.setattr(
            search_mod,
            "search_web",
            lambda *a, **k: SearchResponse(query="q", error="network down"),
        )
        monkeypatch.setattr(
            "auralith.graph.nodes.search_web",
            lambda *a, **k: SearchResponse(query="q", error="network down"),
        )
        out = search_node({"query": "q", "parameters": "q"}, FakeChatModel("unused"))
        assert out["error"] == "network down"
        assert "could not retrieve" in out["answer"].lower()

    def test_results_reach_state_and_are_synthesised(self, monkeypatch):
        """The core fix versus pyautogui: results become data, not keystrokes."""
        from auralith.tools.search import SearchResponse, SearchResult

        monkeypatch.setattr(
            "auralith.graph.nodes.search_web",
            lambda *a, **k: SearchResponse(
                query="q",
                results=[SearchResult("T", "https://example.com", "snippet text")],
            ),
        )
        out = search_node(
            {"query": "q", "parameters": "q"}, FakeChatModel("Synthesised answer [1]")
        )
        assert out["search_results"][0]["url"] == "https://example.com"
        assert out["answer"] == "Synthesised answer [1]"


# --------------------------------------------------------------------------
# Recovery and edges
# --------------------------------------------------------------------------

class TestRecovery:
    def test_retries_while_budget_remains(self):
        out = recovery_node({"query": "q", "retry_count": 0})
        assert out["retry_count"] == 1
        assert out.get("route") is None  # not yet forced to converse
        assert recovery_decision({**out, "query": "q"}) == "route"

    def test_falls_back_to_converse_once_budget_is_spent(self):
        out = recovery_node({"query": "q", "retry_count": MAX_RETRIES})
        assert out["route"] == Route.CONVERSE.value
        assert out["error"] is None  # degraded gracefully, not surfaced as failure
        assert recovery_decision(out) == "converse"

    def test_marks_state_as_recovered(self):
        assert recovery_node({"query": "q", "retry_count": 0})["recovered"] is True

    def test_never_calls_a_model(self):
        """Recovery only decides; it must not add latency or a failure point."""
        out = recovery_node({"query": "q", "retry_count": 0})
        assert set(out) <= {"retry_count", "recovered", "trace", "route", "parameters", "error"}


class TestEdgePredicates:
    @pytest.mark.parametrize(
        "route,expected",
        [
            (Route.SEARCH_WEB.value, "search"),
            (Route.WRITE_CODE.value, "code"),
            (Route.CONVERSE.value, "converse"),
            (Route.RECOVER.value, "recover"),
            (None, "recover"),
            ("GarbageAction", "recover"),
        ],
    )
    def test_route_decision_maps_every_case(self, route, expected):
        """Includes unknown and None so the router has no unhandled branch."""
        assert route_decision({"route": route}) == expected


# --------------------------------------------------------------------------
# State plumbing
# --------------------------------------------------------------------------

class TestState:
    def test_initial_state_has_expected_defaults(self):
        s = initial_state("hello")
        assert s["query"] == "hello"
        assert s["retry_count"] == 0
        assert s["recovered"] is False
        assert s["trace"] == []

    def test_trace_reducer_appends(self):
        assert _append(["a"], ["b"]) == ["a", "b"]
        assert _append(None, ["a"]) == ["a"]
        assert _append(["a"], None) == ["a"]

    def test_keep_last_reducer_ignores_none(self):
        """A later node returning None must not wipe an earlier node's value."""
        assert _keep_last("SearchWeb", None) == "SearchWeb"
        assert _keep_last("SearchWeb", "Converse") == "Converse"
        assert _keep_last(None, "Converse") == "Converse"

    def test_reducers_are_the_ones_wired_into_the_state(self):
        """Guards against the tests drifting from the actual annotations."""
        hints = get_type_hints(AgentState, include_extras=True)
        assert hints["trace"].__metadata__[0] is _append
        assert hints["route"].__metadata__[0] is _keep_last
