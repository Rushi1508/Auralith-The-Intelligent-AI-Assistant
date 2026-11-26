"""Typed state for the agent graph.

The inherited chat_module kept its working data in local variables inside one
128-line function, and its control flow in an if/elif chain over a dict key that
could raise before the chain was ever reached:

    data = output_parser.parse(output)      # raises on malformed JSON
    if data['action'] == 'SearchWeb': ...   # KeyError if the key is absent

Making the state explicit buys three things that matter here:

* Every field a node may read or write is declared in one place, so adding a
  node does not mean auditing a long function for hidden dependencies.
* Failure is part of the state (`error`, `retry_count`), not an exception that
  unwinds the whole run. That is what makes a recovery edge expressible.
* The graph can be inspected, drawn and unit-tested as a transition system
  rather than exercised end-to-end through Streamlit.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class Route(StrEnum):
    """The actions the router may choose.

    Same three actions the prototype had, plus RECOVER, which it lacked - and
    lacking it is precisely why a malformed model response killed the request.
    """

    SEARCH_WEB = "SearchWeb"
    WRITE_CODE = "WriteCode"
    CONVERSE = "Converse"
    RECOVER = "Recover"

    @classmethod
    def parse(cls, raw: str | None) -> "Route | None":
        """Best-effort parse of a model-produced action label.

        Tolerant on purpose: small models return "search_web", "SEARCH WEB",
        or a sentence containing the word. Being strict here would push work
        onto the recovery path that does not belong there. Returns None when
        nothing matches, which is the signal to actually recover.
        """
        if not raw:
            return None
        normalised = "".join(ch for ch in raw.lower() if ch.isalnum())
        for route in cls:
            if "".join(ch for ch in route.value.lower() if ch.isalnum()) == normalised:
                return route
        for route in cls:
            if route is cls.RECOVER:
                continue
            key = "".join(ch for ch in route.value.lower() if ch.isalnum())
            if key in normalised:
                return route
        return None


def _keep_last(current: Any, update: Any) -> Any:
    """Reducer: last write wins, but never overwrite with None.

    Lets a recovery node re-run the router without wiping fields an earlier
    partial attempt already populated.
    """
    return current if update is None else update


def _append(current: list[str] | None, update: list[str] | None) -> list[str]:
    """Reducer: concatenate trace fragments across nodes."""
    return (current or []) + (update or [])


class AgentState(TypedDict, total=False):
    """State threaded through every node."""

    # --- input ---
    query: str
    messages: Annotated[list[AnyMessage], add_messages]
    """Chat history; only carries across turns when a checkpointer is used."""

    # --- routing ---
    route: Annotated[str | None, _keep_last]
    route_raw: Annotated[str | None, _keep_last]
    """Unparsed router output, kept for debugging a bad classification."""
    parameters: Annotated[str | None, _keep_last]

    # --- outputs, one per terminal node ---
    answer: Annotated[str | None, _keep_last]
    code: Annotated[str | None, _keep_last]
    language: Annotated[str | None, _keep_last]
    search_results: Annotated[list[dict[str, Any]] | None, _keep_last]

    # --- failure handling ---
    error: Annotated[str | None, _keep_last]
    retry_count: Annotated[int, _keep_last]
    recovered: Annotated[bool, _keep_last]
    """True when the answer came from the recovery path, not a clean route."""

    # --- trace ---
    trace: Annotated[list[str], _append]
    """Append-only node visit log. Makes graph transitions assertable in tests."""


def initial_state(query: str) -> AgentState:
    return AgentState(
        query=query,
        route=None,
        route_raw=None,
        parameters=None,
        answer=None,
        code=None,
        language=None,
        search_results=None,
        error=None,
        retry_count=0,
        recovered=False,
        trace=[],
    )
