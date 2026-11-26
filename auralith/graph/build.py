"""Graph assembly.

    START
      |
      v
    route ──(SearchWeb)──> search ───┐
      |  ──(WriteCode)───> code ─────┤
      |  ──(Converse)────> converse ─┼──> END
      |  ──(unparseable)─> recover   │
      |                      |       │
      |<──(retry budget left)┘       │
                             |       │
                             └──(budget spent)──> converse

The cycle `route -> recover -> route` is the reason this is a graph and not a
chain. A linear LCEL pipeline cannot express "go back and try again", so the
prototype's only option on a malformed classification was to raise. The retry
budget in recovery_node is what keeps that cycle finite.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite

from auralith.config import Settings
from auralith.graph.nodes import make_nodes, recovery_decision, route_decision
from auralith.graph.state import AgentState, initial_state

log = logging.getLogger(__name__)

# shortcut: unbounded per-session growth, replace with TTL/Postgres-only in a later phase
_MEMORY = InMemorySaver()


def build_graph(settings: Settings, compile_graph: bool = True, checkpointer=None):
    """Build the agent graph.

    `compile_graph=False` returns the uncompiled StateGraph, which tests use to
    inspect the topology without needing a model backend.
    """
    nodes = make_nodes(settings)
    graph = StateGraph(AgentState)

    for name, fn in nodes.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route")

    # Explicit mapping rather than bare string returns: the set of legal
    # destinations is declared here, so a typo in a predicate fails at build
    # time instead of at 2am on an unusual query.
    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "search": "search",
            "code": "code",
            "converse": "converse",
            "recover": "recover",
        },
    )
    graph.add_conditional_edges(
        "recover",
        recovery_decision,
        {"route": "route", "converse": "converse"},
    )

    for terminal in ("search", "code", "converse"):
        graph.add_edge(terminal, END)

    return graph.compile(checkpointer=checkpointer) if compile_graph else graph


@functools.lru_cache
def _pg_setup(url: str) -> None:
    from langgraph.checkpoint.postgres import PostgresSaver

    import psycopg

    # Serialise setup across workers; concurrent migrations can collide.
    with psycopg.connect(url, autocommit=True) as lock:
        lock.execute("SELECT pg_advisory_lock(4243)")
        try:
            with PostgresSaver.from_conn_string(url) as saver:
                saver.setup()
        finally:
            lock.execute("SELECT pg_advisory_unlock(4243)")


def _invoke(app, query: str, config: dict[str, Any]) -> AgentState:
    state = initial_state(query)
    # _keep_last ignores None and trace appends, so on a reused thread the
    # previous turn's code/search_results would leak and trace would accumulate.
    # Overwrite only on a thread with saved state: langgraph does not unwrap it
    # on a channel's first write.
    if app.checkpointer and app.get_state(config).values:
        fresh = {k: Overwrite(v) for k, v in state.items() if k != "query"}
        state = {**fresh, "query": query}
    return app.invoke({**state, "messages": [HumanMessage(query)]}, config=config)


def run_graph(
    query: str,
    settings: Settings,
    thread_id: str | None = None,
    recursion_limit: int = 12,
) -> AgentState:
    """Run one query through the graph.

    `recursion_limit` is a hard stop on top of the retry budget. Two independent
    limits is deliberate: the budget is the intended mechanism, and the
    recursion limit is the backstop for a bug in the budget.
    """
    config: dict[str, Any] = {"recursion_limit": recursion_limit}
    if thread_id:
        config["configurable"] = {"thread_id": thread_id}
    try:
        if thread_id and settings.database_url:
            from langgraph.checkpoint.postgres import PostgresSaver

            _pg_setup(settings.database_url)
            with PostgresSaver.from_conn_string(settings.database_url) as saver:
                app = build_graph(settings, checkpointer=saver)
                return _invoke(app, query, config)
        app = build_graph(settings, checkpointer=_MEMORY if thread_id else None)
        return _invoke(app, query, config)
    except Exception as exc:  # noqa: BLE001
        log.error("Graph execution failed for %r: %s", query[:60], exc)
        state = initial_state(query)
        state["error"] = f"{type(exc).__name__}: {exc}"
        state["answer"] = (
            "Something went wrong handling that request. Details: "
            f"{type(exc).__name__}: {exc}"
        )
        state["trace"] = ["graph:failed"]
        return state


def describe_graph(settings: Settings) -> str:
    """Render the compiled graph as Mermaid, for the README diagram."""
    try:
        return build_graph(settings).get_graph().draw_mermaid()
    except Exception as exc:  # noqa: BLE001
        return f"(could not render graph: {exc})"


def summarise(state: AgentState) -> dict[str, Any]:
    """Flatten a final state into something printable."""
    return {
        "route": state.get("route"),
        "answer": state.get("answer"),
        "recovered": state.get("recovered", False),
        "retries": state.get("retry_count", 0),
        "error": state.get("error"),
        "trace": " -> ".join(state.get("trace", [])),
        "n_search_results": len(state.get("search_results") or []),
    }
