"""Graph nodes.

Each node is a plain function `AgentState -> dict` returning only the keys it
changes. Keeping them free of graph machinery means every one can be unit-tested
by calling it with a dict, which is most of what makes the routing logic
testable at all - the prototype's equivalent logic was unreachable without
standing up Streamlit and a live OpenAI key.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from auralith.config import Settings
from auralith.graph.state import AgentState, Route
from auralith.providers import get_chat_model
from auralith.tools.search import search_web

log = logging.getLogger(__name__)

MAX_RETRIES = 2
# shortcut: last N raw messages, no summarization; upgrade when long threads matter
HISTORY_MESSAGES = 6

ROUTER_PROMPT = """You are a request router. Classify the user's request into \
exactly one action.

Actions:
- SearchWeb: the answer depends on current or changing information that must \
be looked up: news, prices, weather, schedules, latest versions, local places.
- WriteCode: the user wants code as the answer - write, fix, convert, or show \
an example of code. Pick this even if the request is phrased as a question.
- Converse: anything you can answer yourself - explanation (including \
explaining code), discussion, advice, translation, maths, writing prose.

Do not choose SearchWeb just because a request mentions a technology or asks \
"how to"; choose it only when the answer could be out of date.

Respond with ONLY a JSON object and nothing else:
{{"action": "SearchWeb" | "WriteCode" | "Converse", "parameters": "<the search \
query, the code request, or the user's question>"}}

{history}User request: {query}"""

CODE_PROMPT = """You are Auralith, a coding assistant. Write code for the request \
below.

Output ONLY the code, inside a single fenced block that names the language. \
No explanation before or after.

{history}Request: {query}"""

CONVERSE_PROMPT = """You are Auralith, a helpful assistant. Answer the user's \
request clearly. If it is a task, give concrete step-by-step instructions.

{history}Request: {query}"""

SEARCH_SYNTHESIS_PROMPT = """Answer the user's question using the web search \
results below. Cite sources by their number, like [1].

If the results do not answer the question, say so plainly.

Search results:
{results}

{history}Question: {query}

Answer:"""


def _extract_json(text: str) -> dict[str, Any] | None:
    """Recover a JSON object from a possibly-chatty model response."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else text
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _history(state: AgentState) -> str:
    """Prior turns as prompt text; empty without a checkpointer or on turn one."""
    prior = (state.get("messages") or [])[:-1][-HISTORY_MESSAGES:]
    if not prior:
        return ""
    who = {"human": "User"}
    lines = [f"{who.get(m.type, 'Assistant')}: {m.content}" for m in prior]
    return "Conversation so far:\n" + "\n".join(lines) + "\n\n"


def _content(response: Any) -> str:
    return (response.content if hasattr(response, "content") else str(response)).strip()


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def route_node(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    """Classify the request.

    Returns a state update rather than raising. Where the prototype called
    `output_parser.parse(output)` and let a JSONDecodeError propagate out of the
    request, an unparseable response here sets route=Recover and lets the graph
    decide what to do about it.
    """
    query = state["query"]
    try:
        prompt = ROUTER_PROMPT.format(history=_history(state), query=query)
        raw = _content(llm.invoke(prompt))
    except Exception as exc:  # noqa: BLE001
        log.warning("Router call failed: %s", exc)
        return {
            "route": Route.RECOVER.value,
            "error": f"Router call failed: {type(exc).__name__}: {exc}",
            "trace": ["route:error"],
        }

    parsed = _extract_json(raw)
    action = Route.parse(parsed.get("action") if parsed else raw)

    if action is None:
        log.info("Router produced unroutable output: %r", raw[:120])
        return {
            "route": Route.RECOVER.value,
            "route_raw": raw[:500],
            "error": "Router output could not be parsed into a known action.",
            "trace": ["route:unparseable"],
        }

    parameters = (parsed or {}).get("parameters") or query
    return {
        "route": action.value,
        "route_raw": raw[:500],
        "parameters": str(parameters),
        "error": None,
        "trace": [f"route:{action.value}"],
    }


def search_node(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    """Search the web, then synthesise an answer from the results.

    The prototype's search returned "Done!" and the results only ever existed
    on the user's screen. Here they come back into state and are fed to the
    model, which is the difference between a tool call and a side effect.
    """
    query = state.get("parameters") or state["query"]
    response = search_web(query)

    results = [
        {"title": r.title, "url": r.url, "snippet": r.snippet} for r in response.results
    ]
    if not response.ok:
        return {
            "search_results": results,
            "error": response.error or "Search returned no results.",
            "answer": (
                f"I could not retrieve web results for {query!r}. "
                f"({response.error or 'no results'})"
            ),
            "trace": ["search:failed"],
        }

    try:
        answer = _content(
            llm.invoke(
                SEARCH_SYNTHESIS_PROMPT.format(
                    history=_history(state), results=response.format(), query=query
                )
            )
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "search_results": results,
            "error": f"Synthesis failed: {exc}",
            "answer": "Found results but could not summarise them:\n\n"
            + response.format(),
            "trace": ["search:synthesis-failed"],
        }

    return {
        "search_results": results,
        "answer": answer,
        "error": None,
        "trace": [f"search:ok({len(results)})"],
    }


def code_node(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    """Generate code and strip the surrounding fence.

    The prototype popped the first and last lines if they started with a
    backtick, which silently deletes a real line of code whenever the model
    omits the closing fence. A regex over the whole block is both safer and
    recovers the language tag as a bonus.
    """
    query = state.get("parameters") or state["query"]
    try:
        prompt = CODE_PROMPT.format(history=_history(state), query=query)
        raw = _content(llm.invoke(prompt))
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Code generation failed: {type(exc).__name__}: {exc}",
            "trace": ["code:error"],
        }

    fence = re.search(r"```([a-zA-Z0-9+#-]*)\n(.*?)```", raw, re.DOTALL)
    if fence:
        language, code = fence.group(1) or "text", fence.group(2).strip()
    else:
        language, code = "text", raw

    return {
        "code": code,
        "language": language,
        "answer": f"```{language}\n{code}\n```",
        "error": None,
        "trace": [f"code:{language}"],
    }


def converse_node(state: AgentState, llm: BaseChatModel) -> dict[str, Any]:
    query = state.get("parameters") or state["query"]
    try:
        prompt = CONVERSE_PROMPT.format(history=_history(state), query=query)
        answer = _content(llm.invoke(prompt))
    except Exception as exc:  # noqa: BLE001
        return {
            "error": f"Generation failed: {type(exc).__name__}: {exc}",
            "trace": ["converse:error"],
        }
    return {"answer": answer, "error": None, "trace": ["converse:ok"]}


def recovery_node(state: AgentState) -> dict[str, Any]:
    """Handle a routing failure.

    Two-stage, and the staging is the point:

    1. Under the retry budget, bump `retry_count` and send control back to the
       router. Small models fail to emit clean JSON intermittently, so a plain
       retry fixes most of these.
    2. Budget exhausted - degrade to Converse rather than surfacing an error.
       A request the router could not classify is still a request the model can
       usually answer; refusing to answer because a classifier hiccuped is a
       worse outcome for the user than answering conversationally.

    Note this node never calls a model itself. It only decides.
    """
    attempts = state.get("retry_count", 0)
    if attempts < MAX_RETRIES:
        return {
            "retry_count": attempts + 1,
            "recovered": True,
            "trace": [f"recover:retry({attempts + 1}/{MAX_RETRIES})"],
        }
    return {
        "route": Route.CONVERSE.value,
        "parameters": state["query"],
        "retry_count": attempts,
        "recovered": True,
        "error": None,
        "trace": ["recover:fallback-to-converse"],
    }


# --------------------------------------------------------------------------
# Edge predicates
# --------------------------------------------------------------------------

def route_decision(state: AgentState) -> str:
    """Conditional edge out of the router."""
    route = state.get("route")
    if route == Route.SEARCH_WEB.value:
        return "search"
    if route == Route.WRITE_CODE.value:
        return "code"
    if route == Route.CONVERSE.value:
        return "converse"
    return "recover"


def recovery_decision(state: AgentState) -> str:
    """Conditional edge out of recovery: retry the router, or give up gracefully.

    Reads `route` rather than `retry_count` because recovery_node sets route to
    Converse exactly when it has decided to stop retrying. Keeping the decision
    in one place stops the two functions from disagreeing about the budget.
    """
    return "converse" if state.get("route") == Route.CONVERSE.value else "route"


def make_nodes(settings: Settings) -> dict[str, Any]:
    """Bind a model into each node so the graph builder stays declarative."""
    llm = get_chat_model(settings)

    def replying(fn):
        """Append the answer to `messages` so the next turn sees it."""

        def node(s: AgentState) -> dict[str, Any]:
            out = fn(s, llm)
            if out.get("answer"):
                out["messages"] = [AIMessage(out["answer"])]
            return out

        return node

    return {
        "route": lambda s: route_node(s, llm),
        "search": replying(search_node),
        "code": replying(code_node),
        "converse": replying(converse_node),
        "recover": recovery_node,
    }
