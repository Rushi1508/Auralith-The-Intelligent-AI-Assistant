"""Router evaluation (Phase G).

Measures how often the agent's router (`route_node`) sends a request to the
right action, on the hand-labelled set in eval/routes.json.

Only the routing step runs: no web search, no code or answer generation. That
keeps a full pass to one short model call per request (plus retries), so it
takes minutes, not the hour an end-to-end RAG run takes.

Two routes are recorded per request:
  * first: the router's first answer, with Recover meaning unparseable output.
  * final: what the graph would actually execute after the recovery loop
    (retry up to MAX_RETRIES, then fall back to Converse).
Accuracy is scored on `final`, because that is what the user gets.

Usage:
    py eval/eval_router.py                       # configured generator
    py eval/eval_router.py --model qwen2.5:7b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auralith.config import Settings  # noqa: E402
from auralith.graph.nodes import MAX_RETRIES, route_node  # noqa: E402
from auralith.graph.state import Route, initial_state  # noqa: E402
from auralith.providers import get_chat_model  # noqa: E402
from eval.metrics import bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ACTIONS = [Route.SEARCH_WEB.value, Route.WRITE_CODE.value, Route.CONVERSE.value]


def route_one(query: str, llm: Any) -> dict[str, Any]:
    """Run the router and the graph's recovery policy, without the terminal nodes."""
    state = initial_state(query)
    first = route_node(state, llm)
    update, retries = first, 0
    while update["route"] == Route.RECOVER.value and retries < MAX_RETRIES:
        retries += 1
        update = route_node(state, llm)
    final = update["route"]
    if final == Route.RECOVER.value:
        final = Route.CONVERSE.value  # mirrors recovery_node's fallback
    return {
        "first": first["route"],
        "final": final,
        "retries": retries,
        "raw": update.get("route_raw"),
    }


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy (with 95% bootstrap CI), per-kind accuracy and confusion matrix."""
    correct = [1.0 if r["final"] == r["label"] else 0.0 for r in rows]
    out: dict[str, Any] = {
        "n": len(rows),
        "accuracy": round(sum(correct) / len(rows), 4) if rows else 0.0,
        "accuracy_ci95": bootstrap_ci(correct),
        "first_attempt_unparseable": sum(r["first"] == Route.RECOVER.value for r in rows),
        "by_kind": {},
        # confusion[label][predicted] = count
        "confusion": {a: {b: 0 for b in ACTIONS} for a in ACTIONS},
    }
    for kind in sorted({r["kind"] for r in rows}):
        sub = [c for c, r in zip(correct, rows) if r["kind"] == kind]
        out["by_kind"][kind] = {"n": len(sub), "accuracy": round(sum(sub) / len(sub), 4)}
    for r in rows:
        out["confusion"][r["label"]][r["final"]] += 1
    return out


def to_markdown(summary: dict[str, Any], model: str) -> str:
    lo, hi = summary["accuracy_ci95"]
    lines = [
        f"## Router accuracy: {model}",
        "",
        f"- accuracy: **{summary['accuracy']:.3f}** ({summary['n']} requests), 95% CI [{lo:.3f}, {hi:.3f}]",
        *(f"- {k}: {v['accuracy']:.3f} (n={v['n']})" for k, v in summary["by_kind"].items()),
        f"- unparseable on first attempt: {summary['first_attempt_unparseable']}",
        "",
        "| label \\ predicted | " + " | ".join(ACTIONS) + " |",
        "|---|" + "---|" * len(ACTIONS),
    ]
    for label in ACTIONS:
        row = summary["confusion"][label]
        lines.append(f"| {label} | " + " | ".join(str(row[p]) for p in ACTIONS) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="router model (default: configured generator)")
    parser.add_argument("--routes", default=str(ROOT / "eval" / "routes.json"))
    parser.add_argument("--out", help="result file (default: results/router_<model>.json)")
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="seconds to sleep between requests (hosted free tiers rate-limit with 429)",
    )
    args = parser.parse_args()

    settings = Settings.from_env()
    model = args.model or settings.generator_model
    llm = get_chat_model(settings, model=model)
    items = json.loads(Path(args.routes).read_text(encoding="utf-8"))["routes"]

    rows = []
    for i, item in enumerate(items, 1):
        if args.delay and i > 1:
            time.sleep(args.delay)
        t0 = time.perf_counter()
        result = route_one(item["query"], llm)
        rows.append({**item, **result, "seconds": round(time.perf_counter() - t0, 2)})
        mark = "ok " if result["final"] == item["label"] else "XX "
        print(f"[{i:2}/{len(items)}] {mark}{item['id']}: {item['label']} -> {result['final']}", flush=True)

    summary = score(rows)
    out = Path(args.out or ROOT / "results" / f"router_{model.replace(':', '_').replace('/', '_')}.json")
    out.write_text(
        json.dumps(
            {
                "model": model,
                # Mirrors get_chat_model: only the generator model goes hosted.
                "provider": (
                    "gemini"
                    if settings.generator_provider == "gemini"
                    and model == settings.generator_model
                    else settings.provider
                ),
                "temperature": settings.temperature,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(to_markdown(summary, model))
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
