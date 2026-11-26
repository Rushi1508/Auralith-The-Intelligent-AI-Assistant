"""Web search as a real tool.

Replaces `macros.searchWeb`, which did this:

    openChrome()
    time.sleep(0.5)
    pyautogui.write(data)
    pyautogui.press("enter")

That is keystroke injection, not a tool. It typed model-generated text into
whatever window happened to have focus after an arbitrary half-second sleep,
returned the string "Done!" regardless of what happened, and produced no data
the program could use. It could not fail meaningfully, could not be tested, and
could not feed a result back into the model. It also types into whatever the
user was doing if Chrome was slow to focus.

This returns structured results into graph state, so the model can actually read
what it found. Two backends:

* ddgs (default) - no API key, no account, works out of the box, which keeps the
  repo runnable by anyone who clones it.
* Tavily - better quality and built for LLM use, needs TAVILY_API_KEY.

Failures return an empty result list with an `error` field rather than raising,
because in the graph a failed search should route to a recovery node, not crash
the run.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def format(self, n: int) -> str:
        return f"[{n}] {self.title}\n    {self.url}\n    {self.snippet}"


@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult] = field(default_factory=list)
    backend: str = "ddgs"
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.results)

    def format(self) -> str:
        if self.error:
            return f"Search failed: {self.error}"
        if not self.results:
            return "No results found."
        return "\n".join(r.format(i + 1) for i, r in enumerate(self.results))


def _search_ddgs(query: str, max_results: int) -> SearchResponse:
    try:
        from ddgs import DDGS
    except ImportError as exc:
        return SearchResponse(query, backend="ddgs", error=f"ddgs not installed: {exc}")

    for attempt in (1, 2):  # ddgs rate limits are often transient
        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=max_results))
            break
        except Exception as exc:  # noqa: BLE001 - network/ratelimit/parse all land here
            log.warning("ddgs search failed for %r (attempt %d): %s", query, attempt, exc)
            if attempt == 2:
                return SearchResponse(
                    query, backend="ddgs", error=f"{type(exc).__name__}: {exc}"
                )
            time.sleep(1)

    return SearchResponse(
        query=query,
        backend="ddgs",
        results=[
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href") or r.get("url", ""),
                snippet=(r.get("body") or "")[:400],
            )
            for r in raw
        ],
    )


def _search_tavily(query: str, max_results: int) -> SearchResponse:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return SearchResponse(
            query, backend="tavily", error="TAVILY_API_KEY is not set"
        )
    try:
        from tavily import TavilyClient
    except ImportError as exc:
        return SearchResponse(
            query, backend="tavily", error=f"tavily-python not installed: {exc}"
        )

    try:
        raw = TavilyClient(api_key=api_key).search(query, max_results=max_results)
    except Exception as exc:  # noqa: BLE001
        log.warning("Tavily search failed for %r: %s", query, exc)
        return SearchResponse(query, backend="tavily", error=f"{type(exc).__name__}: {exc}")

    return SearchResponse(
        query=query,
        backend="tavily",
        results=[
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=(r.get("content") or "")[:400],
            )
            for r in raw.get("results", [])
        ],
    )


def search_web(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    backend: str | None = None,
) -> SearchResponse:
    """Run a web search and return structured results. Never raises."""
    if not (query or "").strip():
        return SearchResponse(query=query, error="Empty query")

    chosen = (backend or os.getenv("AURALITH_SEARCH_BACKEND", "ddgs")).lower()
    if chosen == "tavily":
        response = _search_tavily(query, max_results)
        # Falling back keeps the graph working when a key is missing, which is
        # the common case for someone who just cloned the repo.
        if response.error and "TAVILY_API_KEY" in response.error:
            log.info("Tavily unavailable, falling back to ddgs")
            return _search_ddgs(query, max_results)
        return response
    return _search_ddgs(query, max_results)
