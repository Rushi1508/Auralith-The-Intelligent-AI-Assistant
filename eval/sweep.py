"""Configuration sweeps.

Runs a list of `RetrievalConfig`s against the eval set and writes both a JSON
record and a markdown comparison table.

Sweeps default to retrieval-only. That is not a shortcut - it is the point.
recall@k and MRR are computed from chunk metadata alone, with no generator and
no judge, so they are deterministic, fast, and uncontaminated by generation
noise. A config that retrieves the right passage but whose 3B generator garbles
the answer is a *generation* problem, and mixing the two would make the
chunking comparison unreadable. End-to-end answer quality is measured
separately, on the winner, with `--generate`.

Usage:
    python eval/sweep.py --sweep chunking
    python eval/sweep.py --sweep k
    python eval/sweep.py --sweep hybrid
    python eval/sweep.py --sweep rerank
    python eval/sweep.py --sweep multiquery
    python eval/sweep.py --sweep finalists --generate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auralith.config import (  # noqa: E402
    BASELINE_CONFIG,
    RESULTS_DIR,
    RetrievalConfig,
    Settings,
)
from auralith.providers import check_backend  # noqa: E402
from eval.run_eval import load_questions, run_config  # noqa: E402

log = logging.getLogger("sweep")


# --------------------------------------------------------------------------
# Sweep definitions
# --------------------------------------------------------------------------

def sweep_chunking() -> list[RetrievalConfig]:
    """Chunk size x overlap, k held at the baseline's 4.

    Sizes 500/1000/1500 with 10% and 20% overlap, plus the inherited 7500/100
    setting as the control. Everything else is fixed so any movement is
    attributable to chunking alone.

    `recursive` is used for the new sizes because CharacterTextSplitter refuses
    to split a block that has no blank line in it, which means at small chunk
    sizes it silently emits oversized chunks and the requested size stops
    meaning anything. The baseline keeps `character` so the control is a true
    control.
    """
    configs = [BASELINE_CONFIG.with_(name="baseline-control")]
    for size in (500, 1000, 1500):
        for pct in (0.10, 0.20):
            configs.append(
                RetrievalConfig(
                    name=f"chunk{size}-ov{int(pct * 100)}",
                    splitter="recursive",
                    chunk_size=size,
                    chunk_overlap=int(size * pct),
                    k=4,
                    tags=("phase-b", "chunking"),
                )
            )
    return configs


def sweep_k(base: RetrievalConfig | None = None) -> list[RetrievalConfig]:
    """k sweep on the winning chunk scheme.

    Run *after* chunking, on the chunking winner, because k and chunk size
    interact: smaller chunks mean each one carries less of the answer, so the
    best k shifts upward as chunk size falls. Sweeping them independently and
    combining the winners would be unsound; this is a sequential greedy search,
    which is a real limitation and is stated as one in the README.
    """
    base = base or RetrievalConfig(
        splitter="recursive", chunk_size=1000, chunk_overlap=150
    )
    return [
        base.with_(name=f"{base.name}-k{k}", k=k, tags=("phase-b", "k"))
        for k in (1, 2, 4, 6, 8, 10)
    ]


def sweep_hybrid(base: RetrievalConfig) -> list[RetrievalConfig]:
    """Dense/BM25 ensemble weighting.

    0.0 is BM25-only and 1.0 is dense-only, so the two endpoints are the
    single-retriever controls. Without those endpoints an ensemble that happens
    to beat the dense baseline could not be distinguished from one that is
    simply riding a strong BM25.
    """
    return [base.with_(name=f"{base.name}-dense-only", use_bm25=False)] + [
        base.with_(
            name=f"{base.name}-hybrid{w:.2f}",
            use_bm25=True,
            dense_weight=w,
            tags=("phase-c", "hybrid"),
        )
        for w in (0.0, 0.3, 0.5, 0.7, 1.0)
    ]


def sweep_rerank(base: RetrievalConfig) -> list[RetrievalConfig]:
    """Cross-encoder reranking at several candidate-pool depths.

    fetch_k controls how much material the cross-encoder gets to re-sort. Too
    small and there is nothing to rescue; too large and the reranker is being
    asked to find one passage among mostly noise, which costs latency and can
    hurt. 20 is the value the brief specifies; the neighbours are included so
    the choice is evidenced rather than asserted.
    """
    return [base.with_(name=f"{base.name}-norerank", use_rerank=False)] + [
        base.with_(
            name=f"{base.name}-rr{f}",
            use_rerank=True,
            rerank_fetch_k=f,
            tags=("phase-c", "rerank"),
        )
        for f in (10, 20, 30)
    ]


def sweep_multiquery(base: RetrievalConfig) -> list[RetrievalConfig]:
    """Query expansion on/off.

    Only two configs, because MultiQueryRetriever's rewrite count is not
    directly exposed - it is a property of the generation prompt. The measured
    question is simply whether expansion earns its latency cost at all.
    """
    return [
        base.with_(name=f"{base.name}-nomq", use_multiquery=False),
        base.with_(
            name=f"{base.name}-mq",
            use_multiquery=True,
            tags=("phase-c", "multiquery"),
        ),
    ]


SWEEPS: dict[str, Callable[..., list[RetrievalConfig]]] = {
    "chunking": sweep_chunking,
    "k": sweep_k,
    "hybrid": sweep_hybrid,
    "rerank": sweep_rerank,
    "multiquery": sweep_multiquery,
}


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run_sweep(
    configs: list[RetrievalConfig],
    settings: Settings,
    questions: list[dict[str, Any]],
    generate: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg.name}: {cfg.describe()}")
        started = time.perf_counter()
        try:
            result = run_config(
                cfg, settings, questions, generate=generate, judge_answers=generate
            )
        except Exception as exc:  # noqa: BLE001
            log.error("Config %s failed: %s", cfg.name, exc)
            rows.append({"name": cfg.name, "slug": cfg.slug, "error": str(exc)})
            continue

        summary = result["summary"]
        rows.append(
            {
                "name": cfg.name,
                "slug": cfg.slug,
                "describe": cfg.describe(),
                "config": {
                    "splitter": cfg.splitter,
                    "chunk_size": cfg.chunk_size,
                    "chunk_overlap": cfg.chunk_overlap,
                    "k": cfg.k,
                    "use_bm25": cfg.use_bm25,
                    "dense_weight": cfg.dense_weight,
                    "use_rerank": cfg.use_rerank,
                    "rerank_fetch_k": cfg.rerank_fetch_k,
                    "use_multiquery": cfg.use_multiquery,
                },
                "n_chunks": result["index"]["n_chunks"],
                "index_build_seconds": result["index"]["build_seconds"],
                "index_from_cache": result["index"]["from_cache"],
                "summary": summary,
                "by_type": result["by_type"],
                "sweep_seconds": round(time.perf_counter() - started, 2),
            }
        )
        print(
            f"    recall@{cfg.k}={summary.get(f'recall@{cfg.k}', 0):.4f}  "
            f"mrr={summary.get('mrr', 0):.4f}  "
            f"chunks={result['index']['n_chunks']}"
        )
    return rows


def to_markdown(rows: list[dict[str, Any]], title: str, generated: bool) -> str:
    """Render the comparison table.

    Metric column names are normalised to recall@k / precision@k because k
    varies across a sweep and per-row column names would not line up.
    """
    lines = [
        f"### {title}",
        "",
        f"_{len(rows)} configurations. "
        f"{'End-to-end (generation + LLM judge)' if generated else 'Retrieval-only (no generator, no judge)'}._",
        "",
    ]
    header = ["config", "chunks", "size/ovl", "k", "extras", "recall@k", "hit@k", "MRR", "prec@k"]
    if generated:
        header += ["answer_acc", "abstain_acc"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    best_recall = max(
        (r["summary"].get(f"recall@{r['config']['k']}", 0) for r in rows if "summary" in r),
        default=0,
    )
    best_mrr = max((r["summary"].get("mrr", 0) for r in rows if "summary" in r), default=0)

    for r in rows:
        if "summary" not in r:
            lines.append(f"| {r['name']} | — | — | — | — | FAILED | — | — | — |")
            continue
        c, s = r["config"], r["summary"]
        k = c["k"]
        recall = s.get(f"recall@{k}", 0.0)
        hit = s.get(f"hit@{k}", 0.0)
        precision = s.get(f"precision@{k}", 0.0)
        mrr = s.get("mrr", 0.0)
        extras = []
        if c["use_bm25"]:
            extras.append(f"hybrid {c['dense_weight']:.2f}")
        if c["use_rerank"]:
            extras.append(f"rerank←{c['rerank_fetch_k']}")
        if c["use_multiquery"]:
            extras.append("multiquery")

        cells = [
            r["name"],
            str(r["n_chunks"]),
            f"{c['chunk_size']}/{c['chunk_overlap']}",
            str(c["k"]),
            ", ".join(extras) or "—",
            f"**{recall:.4f}**" if recall == best_recall and recall > 0 else f"{recall:.4f}",
            f"{hit:.4f}",
            f"**{mrr:.4f}**" if mrr == best_mrr and mrr > 0 else f"{mrr:.4f}",
            f"{precision:.4f}",
        ]
        if generated:
            cells += [
                f"{s.get('answer_correctness', 0.0):.4f}",
                f"{s.get('abstention_accuracy', 0.0):.4f}",
            ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True, choices=sorted(SWEEPS) + ["finalists"])
    ap.add_argument(
        "--generate",
        action="store_true",
        help="Also generate answers and judge them (much slower)",
    )
    ap.add_argument("--base", default=None, help="JSON file holding the base config")
    ap.add_argument("--out", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    settings = Settings.from_env()
    check_backend(settings)
    questions = load_questions()

    base = None
    if args.base:
        base = RetrievalConfig(**json.loads(Path(args.base).read_text()))

    if args.sweep == "chunking":
        configs = sweep_chunking()
    elif args.sweep == "k":
        configs = sweep_k(base)
    elif args.sweep == "finalists":
        if not base:
            raise SystemExit("--sweep finalists requires --base")
        configs = [base]
    else:
        if not base:
            raise SystemExit(f"--sweep {args.sweep} requires --base")
        configs = SWEEPS[args.sweep](base)

    print(f"Sweep '{args.sweep}': {len(configs)} configurations, "
          f"{len(questions)} questions, generate={args.generate}")

    rows = run_sweep(configs, settings, questions, generate=args.generate)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / f"sweep_{args.sweep}.json"
    out.write_text(json.dumps({"sweep": args.sweep, "rows": rows}, indent=2), encoding="utf-8")

    md = to_markdown(rows, f"Sweep: {args.sweep}", args.generate)
    md_path = out.with_suffix(".md")
    md_path.write_text(md, encoding="utf-8")

    print("\n" + md)
    print(f"Written to {out} and {md_path}")

    ranked = sorted(
        (r for r in rows if "summary" in r),
        key=lambda r: (
            r["summary"].get(f"recall@{r['config']['k']}", 0),
            r["summary"].get("mrr", 0),
        ),
        reverse=True,
    )
    if ranked:
        winner = ranked[0]
        winner_k = winner["config"]["k"]
        winner_recall = winner["summary"].get(f"recall@{winner_k}", 0.0)
        winner_mrr = winner["summary"].get("mrr", 0.0)
        print(
            f"\nBest by recall then MRR: {winner['name']}  "
            f"(recall@{winner_k}={winner_recall:.4f}, mrr={winner_mrr:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
