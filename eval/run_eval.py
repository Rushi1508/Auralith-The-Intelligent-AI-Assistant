"""Evaluation harness.

Run one `RetrievalConfig` against the eval set and write a result file.

The important structural decision is that retrieval scoring and answer scoring
are separable:

  * recall@k, hit@k, MRR and precision@k depend only on which chunks come back.
    No generator, no judge, no LLM call at all. On this corpus a retrieval-only
    pass over 28 questions takes seconds.
  * answer_correctness needs a generation and a judge call per question, which
    on local models is ~10-20s each.

So sweeps run with `--no-generate` and are cheap enough to be exhaustive, and
only the configs that win on retrieval pay for a full end-to-end run. Coupling
the two would have made the Phase B/C sweeps take hours instead of minutes, and
would have added generator noise to a measurement that does not need it.

Usage:
    python eval/run_eval.py --config baseline --out results/baseline.json
    python eval/run_eval.py --config baseline --no-generate     # retrieval only
    python eval/run_eval.py --validate-judge
    python eval/run_eval.py --rejudge results/baseline.json   # judge only, stored answers
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from auralith.config import (  # noqa: E402
    BASELINE_CONFIG,
    CORPUS_DIR,
    PRODUCTION_CONFIG,
    RESULTS_DIR,
    RetrievalConfig,
    Settings,
)
from auralith.index import build_index, load_chunks_for_bm25  # noqa: E402
from auralith.ingest import load_corpus  # noqa: E402
from auralith.providers import (  # noqa: E402
    check_backend,
    get_judge_model,
    unload_model,
)
from auralith.rag import ANSWER_PROMPT, answer_question  # noqa: E402
from auralith.retrievers import build_retriever, retrieve  # noqa: E402
from eval.judge import (  # noqa: E402
    JUDGE_PROMPT,
    JUDGE_VALIDATION_CASES,
    judge_answer,
    keyword_score,
    load_judge_labels,
    validate_judge,
)
from eval.metrics import (  # noqa: E402
    RelevanceLabel,
    aggregate,
    by_type,
    hit_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_vector,
)

log = logging.getLogger("eval")

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
CHECKPOINT_DIR = RESULTS_DIR / ".checkpoints"


def _generation_tag(settings: Settings) -> str:
    """Short hash of everything besides retrieval that shapes an answer.

    Checkpoints were once keyed on the retrieval config alone, so changing the
    prompt or the generator and re-running silently replayed the old answers
    and reported identical numbers.
    """
    text = ANSWER_PROMPT.messages[0].prompt.template
    key = f"{text}|{settings.generator_model}|{settings.judge_model}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:8]


def _checkpoint_path(config: RetrievalConfig, phase: str) -> Path:
    return CHECKPOINT_DIR / f"{config.slug}__{phase}.jsonl"


def _load_checkpoint(config: RetrievalConfig, phase: str) -> dict[str, dict[str, Any]]:
    """Read completed rows for one phase, keyed by question id.

    A full end-to-end pass costs ~110s per question for generation alone, so a
    28-question run is over an hour. Anything that long needs to survive being
    interrupted, otherwise a single stop discards the whole run - which is
    exactly what happened twice before this existed.

    Corrupt trailing lines are skipped rather than fatal: a run killed
    mid-write leaves a partial final line, and losing one question is much
    better than losing the file.
    """
    path = _checkpoint_path(config, phase)
    if not path.exists():
        return {}
    done: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            log.warning("Skipping corrupt checkpoint line in %s", path.name)
            continue
        done[row["id"]] = row
    return done


def _append_checkpoint(
    config: RetrievalConfig, phase: str, row: dict[str, Any]
) -> None:
    path = _checkpoint_path(config, phase)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
        fh.flush()


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"]


def run_config(
    config: RetrievalConfig,
    settings: Settings,
    questions: list[dict[str, Any]],
    generate: bool = True,
    judge_answers: bool = True,
    corpus_dir: Path = CORPUS_DIR,
    resume: bool = True,
) -> dict[str, Any]:
    """Evaluate one config. Returns a fully self-describing result dict.

    With `resume=True` (default) each completed question is checkpointed to
    results/.checkpoints/ and re-runs skip work already done, so an interrupted
    hour-long evaluation continues instead of starting over.
    """
    started = time.perf_counter()
    docs = load_corpus(corpus_dir)
    store, build_report = build_index(docs, config, settings)
    bm25_chunks = load_chunks_for_bm25(docs, config) if config.use_bm25 else None
    retriever = build_retriever(store, config, settings, bm25_chunks=bm25_chunks)

    tag = _generation_tag(settings)
    phase = f"gen_{tag}" if generate else "retrieval"
    # Judge prompt is hashed separately so a judge change re-judges without regenerating.
    judge_phase = f"judge_{tag}_{hashlib.sha1(JUDGE_PROMPT.encode('utf-8')).hexdigest()[:8]}"
    completed = _load_checkpoint(config, phase) if resume else {}
    if completed:
        log.info("Resuming %s: %d/%d already done", config.name, len(completed), len(questions))

    rows: list[dict[str, Any]] = []
    for i, q in enumerate(questions, 1):
        if q["id"] in completed:
            rows.append(completed[q["id"]])
            continue

        label = RelevanceLabel.from_question(q)
        row: dict[str, Any] = {
            "id": q["id"],
            "question": q["question"],
            "type": q.get("type", "unknown"),
            "answerable": q.get("answerable", True),
            "doc_id": q.get("doc_id"),
            "relevant_pages": q.get("relevant_pages", []),
        }

        if generate:
            result = answer_question(q["question"], retriever, config, settings)
            docs_out = result.passages
            row.update(
                {
                    "answer": result.answer,
                    "abstained": result.abstained,
                    "latency_seconds": result.latency_seconds,
                    "error": result.error,
                    "citations": result.citations,
                }
            )
        else:
            t0 = time.perf_counter()
            try:
                docs_out = retrieve(retriever, q["question"], config)
                err = None
            except Exception as exc:  # noqa: BLE001
                log.warning("Retrieval failed on %s: %s", q["id"], exc)
                docs_out, err = [], f"{type(exc).__name__}: {exc}"
            row.update(
                {
                    "latency_seconds": round(time.perf_counter() - t0, 3),
                    "error": err,
                }
            )

        hits = relevance_vector(docs_out, label)
        row.update(
            {
                "retrieved": [
                    {
                        "doc_id": d.metadata.get("doc_id"),
                        "page": d.metadata.get("page"),
                        "relevant": h,
                    }
                    for d, h in zip(docs_out, hits)
                ],
                "n_retrieved": len(docs_out),
                "recall": recall_at_k(hits, label, config.k),
                "hit": hit_at_k(hits, config.k),
                "reciprocal_rank": reciprocal_rank(hits),
                "precision": precision_at_k(hits, config.k),
            }
        )

        rows.append(row)
        if resume:
            _append_checkpoint(config, phase, row)
        log.info(
            "[%2d/%d] %-11s recall=%.2f rr=%.2f %.0fs",
            i,
            len(questions),
            q["id"],
            row["recall"],
            row["reciprocal_rank"],
            row.get("latency_seconds", 0.0),
        )

    # --- Judging runs as a separate pass, after all generation is finished ---
    #
    # Not a stylistic choice. Interleaving generate/judge per question keeps
    # both models resident: 7.6 GB of weights on a 15.6 GB machine left ~1.1 GB
    # free and pushed the box into paging, and a 28-question run had not
    # finished after 57 minutes. Evicting the generator before loading the judge
    # means only one model is in memory at a time.
    if generate and judge_answers:
        unload_model(settings, settings.generator_model)
        judged = _load_checkpoint(config, judge_phase) if resume else {}
        judge = None
        for i, (row, q) in enumerate(zip(rows, questions), 1):
            if q["id"] in judged:
                row.update(
                    {
                        key: judged[q["id"]][key]
                        for key in (
                            "judge_score",
                            "judge_reason",
                            "judge_parse_failed",
                            "keyword_score",
                        )
                        if key in judged[q["id"]]
                    }
                )
                continue
            # Constructed lazily so a fully-resumed judge phase never loads the
            # 5 GB judge model at all.
            if judge is None:
                judge = get_judge_model(settings)
            verdict = judge_answer(
                judge,
                q["question"],
                q["answer"],
                row.get("answer", ""),
                answerable=q.get("answerable", True),
            )
            row["judge_score"] = verdict.score
            row["judge_reason"] = verdict.reason
            row["judge_parse_failed"] = verdict.parse_failed
            row["keyword_score"] = keyword_score(
                row.get("answer", ""), q.get("answer_keys", [])
            )
            if resume:
                _append_checkpoint(
                    config,
                    judge_phase,
                    {
                        "id": q["id"],
                        "judge_score": row["judge_score"],
                        "judge_reason": row["judge_reason"],
                        "judge_parse_failed": row["judge_parse_failed"],
                        "keyword_score": row["keyword_score"],
                    },
                )
            log.info(
                "[judge %2d/%d] %-11s score=%.0f", i, len(rows), q["id"], verdict.score
            )
        if judge is not None:
            unload_model(settings, settings.judge_model)

    summary = aggregate(rows, config.k)
    return {
        "config": {**asdict(config), "slug": config.slug, "describe": config.describe()},
        "settings": {
            "provider": settings.provider,
            "generator_model": settings.generator_model if generate else None,
            "judge_model": settings.judge_model if (generate and judge_answers) else None,
            "embedding_model": settings.embedding_model,
            "cross_encoder_model": (
                settings.cross_encoder_model if config.use_rerank else None
            ),
            "temperature": settings.temperature,
        },
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "generated_answers": generate,
            "judged": bool(generate and judge_answers),
            "wall_seconds": round(time.perf_counter() - started, 2),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "index": asdict(build_report),
        "summary": summary,
        "by_type": by_type(rows, config.k),
        "questions": rows,
    }


def rejudge(path: Path, settings: Settings) -> dict[str, Any]:
    """Re-score the stored answers in a result file with the current judge.

    Generation checkpoints are keyed on the answer prompt, so runs made with an
    older prompt cannot be resumed; their answers survive only in the result JSON.
    """
    result = json.loads(path.read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in load_questions()}
    judge = get_judge_model(settings)
    rows = result["questions"]
    for i, row in enumerate(rows, 1):
        q = questions[row["id"]]
        verdict = judge_answer(
            judge,
            q["question"],
            q["answer"],
            row.get("answer", ""),
            answerable=q.get("answerable", True),
        )
        row["judge_score"] = verdict.score
        row["judge_reason"] = verdict.reason
        row["judge_parse_failed"] = verdict.parse_failed
        row["keyword_score"] = keyword_score(row.get("answer", ""), q.get("answer_keys", []))
        log.info("[judge %2d/%d] %-11s score=%.0f", i, len(rows), row["id"], verdict.score)
    unload_model(settings, settings.judge_model)
    k = result["config"]["k"]
    result["summary"] = aggregate(rows, k)
    result["by_type"] = by_type(rows, k)
    result["settings"]["judge_model"] = settings.judge_model
    result["run"]["rejudged_utc"] = datetime.now(timezone.utc).isoformat()
    result["run"]["judge_prompt_sha1"] = hashlib.sha1(JUDGE_PROMPT.encode("utf-8")).hexdigest()[:8]
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--rejudge",
        type=Path,
        default=None,
        help="Re-score the stored answers in this result file with the current judge (overwrites it)",
    )
    ap.add_argument("--config", default="baseline", help="Named config to run")
    ap.add_argument("--out", default=None, help="Output JSON path")
    ap.add_argument(
        "--no-generate",
        action="store_true",
        help="Retrieval metrics only; skips generation and judging (fast)",
    )
    ap.add_argument("--no-judge", action="store_true", help="Generate but do not judge")
    ap.add_argument(
        "--validate-judge",
        action="store_true",
        help="Score the judge against hand labels and exit",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("eval").setLevel(logging.INFO)

    settings = Settings.from_env()
    check_backend(settings)

    if args.validate_judge:
        judge = get_judge_model(settings)
        reports = {
            "synthetic": validate_judge(judge, JUDGE_VALIDATION_CASES),
            "real": validate_judge(judge, load_judge_labels()),
        }
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out = RESULTS_DIR / "judge_validation.json"
        out.write_text(
            json.dumps({"judge_model": settings.judge_model, **reports}, indent=2),
            encoding="utf-8",
        )
        for name, report in reports.items():
            print(
                f"Judge agreement ({name}): {report['n_agree']}/"
                f"{report['n_cases']} = {report['agreement']:.1%}"
            )
            for d in report["disagreements"]:
                label = d["id"] or repr(d["candidate"][:60])
                print(f"  DISAGREE {label}: expected={d['expected']} judged={d['judged']}")
                print(f"    {d['reason']}")
        print(f"Written to {out}")
        return 0

    if args.rejudge:
        result = rejudge(args.rejudge, settings)
        args.rejudge.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"answer_correctness: {result['summary'].get('answer_correctness')}")
        print(f"Written to {args.rejudge}")
        return 0

    named = {"baseline": BASELINE_CONFIG, "production": PRODUCTION_CONFIG}
    if args.config not in named:
        raise SystemExit(
            f"Unknown config {args.config!r}. Choose from {sorted(named)}; "
            "arbitrary sweeps are driven by eval/sweep.py."
        )
    config = named[args.config]

    questions = load_questions()
    print(f"Config:    {config.name} - {config.describe()}")
    print(f"Questions: {len(questions)}")
    print(f"Provider:  {settings.provider} ({settings.generator_model})\n")

    result = run_config(
        config,
        settings,
        questions,
        generate=not args.no_generate,
        judge_answers=not args.no_judge,
    )

    out_path = Path(args.out) if args.out else RESULTS_DIR / f"{config.name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n=== SUMMARY ===")
    for key, val in result["summary"].items():
        print(f"  {key:34s} {val}")
    print(f"\nWritten to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
