"""Compare two eval runs with paired bootstrap confidence intervals.

    python eval/compare.py results/baseline.json results/production.json

Prints a markdown table: each metric's mean in both runs, the delta, and a 95%
CI on the delta. Questions are paired by id, so only questions present in both
runs count. No models are loaded; this reads committed JSON only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.metrics import paired_bootstrap_ci  # noqa: E402

Row = dict[str, Any]

# (label, which questions, per-question value). None = row has no value.
METRICS: list[tuple[str, Callable[[Row], bool], Callable[[Row], float | None]]] = [
    ("recall@k", lambda r: r["answerable"], lambda r: r.get("recall")),
    ("hit@k", lambda r: r["answerable"], lambda r: r.get("hit")),
    ("MRR", lambda r: r["answerable"], lambda r: r.get("reciprocal_rank")),
    ("precision@k", lambda r: r["answerable"], lambda r: r.get("precision")),
    ("answer_correctness (all)", lambda r: True, lambda r: r.get("judge_score")),
    ("answer_correctness (answerable)", lambda r: r["answerable"], lambda r: r.get("judge_score")),
    (
        "abstention_accuracy (unanswerable)",
        lambda r: not r["answerable"],
        lambda r: None if "abstained" not in r else float(r["abstained"]),
    ),
    (
        "false_abstention_rate (answerable)",
        lambda r: r["answerable"],
        lambda r: None if "abstained" not in r else float(r["abstained"]),
    ),
    ("keyword_score", lambda r: True, lambda r: r.get("keyword_score")),
    # Path-eval files (eval/eval_paths.py); RAG files have no "path" key.
    *(
        (
            f"answer_correctness ({p})",
            lambda r, p=p: r.get("path") == p,
            lambda r: r.get("judge_score"),
        )
        for p in ("SearchWeb", "WriteCode", "Converse")
    ),
    (
        "code_passes (WriteCode)",
        lambda r: r.get("path") == "WriteCode",
        lambda r: None if r.get("code_passes") is None else float(r["code_passes"]),
    ),
]


def compare(a: dict[str, Any], b: dict[str, Any]) -> list[dict[str, Any]]:
    rows_a = {q["id"]: q for q in a["questions"]}
    rows_b = {q["id"]: q for q in b["questions"]}
    shared = [i for i in rows_a if i in rows_b]

    out = []
    for label, include, value in METRICS:
        pairs = [
            (value(rows_a[i]), value(rows_b[i]))
            for i in shared
            if include(rows_a[i])
        ]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        xs, ys = [float(x) for x, _ in pairs], [float(y) for _, y in pairs]
        lo, hi = paired_bootstrap_ci(xs, ys)
        out.append(
            {
                "metric": label,
                "n": len(pairs),
                "a": round(sum(xs) / len(xs), 4),
                "b": round(sum(ys) / len(ys), 4),
                "delta": round((sum(ys) - sum(xs)) / len(xs), 4),
                "ci": (lo, hi),
                "excludes_zero": lo > 0 or hi < 0,
            }
        )
    return out


def to_markdown(rows: list[dict[str, Any]], name_a: str, name_b: str) -> str:
    lines = [
        f"| metric | n | {name_a} | {name_b} | delta | 95% CI (paired bootstrap) | |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["ci"]
        mark = "**excludes 0**" if r["excludes_zero"] else "includes 0"
        lines.append(
            f"| {r['metric']} | {r['n']} | {r['a']:.4f} | {r['b']:.4f} "
            f"| {r['delta']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {mark} |"
        )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", type=Path, help="reference run (e.g. results/baseline.json)")
    ap.add_argument("b", type=Path, help="candidate run (e.g. results/production.json)")
    args = ap.parse_args()

    a = json.loads(args.a.read_text(encoding="utf-8"))
    b = json.loads(args.b.read_text(encoding="utf-8"))
    print(to_markdown(compare(a, b), args.a.stem, args.b.stem))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
