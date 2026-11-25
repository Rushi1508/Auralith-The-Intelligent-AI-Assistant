"""Path-quality evaluation (Phase H).

Measures the answer quality of each agent path (SearchWeb, WriteCode, Converse)
after routing. Each request in eval/paths.json goes straight to its own node,
bypassing the router, then the answer is scored by the LLM judge. WriteCode
Python answers are also run against assert snippets.

Differences from production, on purpose:
  * the router is bypassed, so `search_node` searches on the raw user query
    instead of the router's rewritten `parameters`;
  * web results are cached in eval/paths_search_cache.json so reruns compare
    generators on identical evidence.

Usage:
    py eval/eval_paths.py                          # configured generator
    py eval/eval_paths.py --requests eval/paths_heldout.json
    py eval/eval_paths.py --rejudge results/paths_x.json   # re-score only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import auralith.graph.nodes as nodes  # noqa: E402
from auralith.config import Settings  # noqa: E402
from auralith.graph.nodes import code_node, converse_node, search_node  # noqa: E402
from auralith.graph.state import initial_state  # noqa: E402
from auralith.providers import (  # noqa: E402
    check_backend,
    get_chat_model,
    get_judge_model,
)
from auralith.tools.search import SearchResponse, SearchResult, search_web  # noqa: E402
from eval.judge import judge_answer, keyword_score  # noqa: E402
from eval.metrics import bootstrap_ci  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PATHS = ["SearchWeb", "WriteCode", "Converse"]
CACHE_PATH = ROOT / "eval" / "paths_search_cache.json"


def cached_search(
    cache_path: Path | str, live: Callable[..., SearchResponse] = search_web
) -> Callable[..., SearchResponse]:
    """Wrap a search function with an on-disk cache keyed by query.

    Only successful responses are stored, so a rate limit never gets frozen in.
    """
    path = Path(cache_path)

    def search(query: str, *args: Any, **kwargs: Any) -> SearchResponse:
        cache = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        hit = cache.get(query)
        if hit:
            return SearchResponse(
                query=query,
                results=[SearchResult(**r) for r in hit["results"]],
                backend=hit["backend"],
            )
        response = live(query, *args, **kwargs)
        if response.ok:
            cache[query] = {
                "backend": response.backend,
                "results": [
                    {"title": r.title, "url": r.url, "snippet": r.snippet}
                    for r in response.results
                ],
            }
            path.write_text(
                json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8"
            )
        return response

    return search


def run_python(code: str, test: str, timeout: float = 10) -> tuple[bool, str]:
    """Run code followed by assert snippet in a fresh interpreter; pass = exit 0."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "candidate.py"
        script.write_text(code + "\n\n" + test + "\n", encoding="utf-8")
        # shortcut: no OS/network isolation (Windows), relies on the request-set rules;
        # use a container if requests ever come from untrusted users
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=tmp,
                stdin=subprocess.DEVNULL,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, "timeout"
    stderr = proc.stderr.decode("utf-8", errors="replace")
    return proc.returncode == 0, stderr[-500:]


def run_one(item: dict[str, Any], llm: Any) -> dict[str, Any]:
    """Run one request through its path's node (router bypassed)."""
    state = initial_state(item["query"])
    path = item["path"]
    node = {
        "SearchWeb": search_node, "WriteCode": code_node, "Converse": converse_node
    }[path]
    update = node(state, llm)
    out: dict[str, Any] = {"answer": update.get("answer"), "error": update.get("error")}
    if path == "SearchWeb":
        results = update.get("search_results") or []
        text = "\n".join(f"{r['title']} {r['url']} {r['snippet']}" for r in results)
        # Only a failed search counts; a failed synthesis after a good search does not.
        out["search_ok"] = update.get("trace") != ["search:failed"]
        out["results_keyword_score"] = keyword_score(text, item.get("answer_keys", []))
    elif path == "WriteCode":
        code, language = update.get("code"), update.get("language")
        out["code"], out["language"] = code, language
        # code_node returns raw text as language "text" when no fence closes.
        out["truncated"] = language == "text" and (code or "").startswith("```")
        out["code_passes"], out["code_stderr"] = None, None
        if item.get("test") and item.get("language") in ("python", "py"):
            out["code_passes"], out["code_stderr"] = (
                run_python(code, item["test"]) if code else (False, "no code generated")
            )
    return out


def _mean(xs: list[float]) -> float | None:
    return round(sum(xs) / len(xs), 4) if xs else None


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-path and overall summary with 95% bootstrap CIs."""

    def block(sub: list[dict[str, Any]]) -> dict[str, Any]:
        judged = [r["judge_score"] for r in sub if r.get("judge_score") is not None]
        out: dict[str, Any] = {
            "n": len(sub),
            "judge_accuracy": _mean(judged),
            "judge_accuracy_ci95": bootstrap_ci(judged) if judged else None,
            "keyword_score": _mean(
                [r["keyword_score"] for r in sub if r.get("keyword_score") is not None]
            ),
            "judge_parse_failed": sum(bool(r.get("judge_parse_failed")) for r in sub),
            "errors": sum(bool(r.get("error")) for r in sub),
        }
        return out

    summary: dict[str, Any] = {}
    for path in PATHS:
        sub = [r for r in rows if r["path"] == path]
        if not sub:
            continue
        s = block(sub)
        if path == "WriteCode":
            passes = [
                float(r["code_passes"]) for r in sub if r.get("code_passes") is not None
            ]
            s["code_pass_n"] = len(passes)
            s["code_pass_rate"] = _mean(passes)
            s["code_pass_rate_ci95"] = bootstrap_ci(passes) if passes else None
            s["truncated"] = sum(bool(r.get("truncated")) for r in sub)
        if path == "SearchWeb":
            s["search_failed"] = sum(r.get("search_ok") is False for r in sub)
            s["results_keyword_score"] = _mean(
                [r["results_keyword_score"] for r in sub
                 if r.get("results_keyword_score") is not None]
            )
        summary[path] = s
    summary["overall"] = block(rows)
    return summary


def _judge_rows(rows: list[dict[str, Any]], judge: Any) -> None:
    for i, row in enumerate(rows, 1):
        verdict = judge_answer(
            judge,
            row["query"],
            row["reference"],
            row.get("answer") or "",
            strip_citations=row["path"] != "WriteCode",  # `arr[0]` is not a citation
        )
        row["judge_score"] = verdict.score
        row["judge_reason"] = verdict.reason
        row["judge_parse_failed"] = verdict.parse_failed
        row["keyword_score"] = keyword_score(
            row.get("answer") or "", row.get("answer_keys", [])
        )
        print(f"judge [{i:2}/{len(rows)}] {row['id']}: {verdict.score:.0f}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", help="generator model (default: configured generator)")
    parser.add_argument("--requests", default=str(ROOT / "eval" / "paths.json"))
    parser.add_argument(
        "--out", help="result file (default: results/paths_<model>_<requests>.json)"
    )
    parser.add_argument("--rejudge", help="re-score the answers stored in this file")
    parser.add_argument("--delay", type=float, default=0.0, help="seconds between items")
    args = parser.parse_args()

    settings = Settings.from_env()
    check_backend(settings, need_judge=True)
    judge = get_judge_model(settings)

    if args.rejudge:
        meta = json.loads(Path(args.rejudge).read_text(encoding="utf-8"))
        model, rows = meta["model"], meta["questions"]
        out = Path(args.out or args.rejudge)
    else:
        model = args.model or settings.generator_model
        safe = model.replace(":", "_").replace("/", "_")
        # Request-set stem in the name so a held-out run never overwrites main.
        name = f"paths_{safe}_{Path(args.requests).stem}.json"
        out = Path(args.out or ROOT / "results" / name)
        meta = {}
        llm = get_chat_model(settings, model=model)
        # Same name tests/test_graph.py patches.
        nodes.search_web = cached_search(CACHE_PATH)
        items = json.loads(Path(args.requests).read_text(encoding="utf-8"))["requests"]
        rows = []

    def write() -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "model": model,
                    # Mirrors get_chat_model: only the generator model goes hosted.
                    "provider": meta.get("provider") or (
                        "gemini"
                        if settings.generator_provider == "gemini"
                        and model == settings.generator_model
                        else settings.provider
                    ),
                    "judge_model": settings.judge_model,
                    "temperature": settings.temperature,
                    "max_output_tokens": settings.max_output_tokens,
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "search_cache": CACHE_PATH.name,
                    "summary": score(rows),
                    "questions": rows,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    if not args.rejudge:
        for i, item in enumerate(items, 1):
            if args.delay and i > 1:
                time.sleep(args.delay)
            t0 = time.perf_counter()
            result = run_one(item, llm)
            rows.append(
                {
                    "id": item["id"],
                    "path": item["path"],
                    "query": item["query"],
                    "reference": item["reference"],
                    "answer_keys": item.get("answer_keys", []),
                    "answerable": True,
                    **result,
                    "seconds": round(time.perf_counter() - t0, 2),
                }
            )
            print(f"gen [{i:2}/{len(items)}] {item['id']}", flush=True)
        write()  # keep the answers even if judging dies

    _judge_rows(rows, judge)
    write()

    summary = score(rows)
    print(json.dumps(summary, indent=2))
    if summary["overall"]["judge_parse_failed"]:
        failed = summary["overall"]["judge_parse_failed"]
        print(
            f"WARNING: {failed} judge outputs could not be parsed (tunnel dropped?). "
            "Rerun with --rejudge.",
            file=sys.stderr,
        )
    print(f"\nWritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
