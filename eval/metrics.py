"""Retrieval metrics.

Everything rests on one definition: `is_relevant`. A retrieved chunk counts as
relevant iff it comes from a labelled document AND its page is one of the pages
a human verified as answering the question.

This is page-level, not span-level, relevance. That is a deliberate trade-off
and worth being able to defend:

  + It is objective and reproducible. No model is in the loop, so the retrieval
    numbers are not contaminated by the thing being evaluated.
  + It stays valid across every chunking scheme in the sweep. A span-level label
    would have to be re-derived for each chunk size, and comparisons across
    configs would stop being apples-to-apples.
  - It is coarse. A chunk from the right page that happens to miss the exact
    sentence still scores as a hit. Under fine chunking this inflates recall
    slightly relative to a span-level judgement.

The direction of that bias matters when reading the sweep: it flatters small
chunk sizes least (more chunks per page means more ways to retrieve the right
page while missing the sentence), so the recall gains reported in Phase B are
if anything conservative, not inflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from langchain_core.documents import Document


@dataclass(frozen=True)
class RelevanceLabel:
    """Ground truth for one question: which (doc_id, page) pairs answer it."""

    pairs: frozenset[tuple[str, int]]

    @classmethod
    def from_question(cls, q: dict[str, Any]) -> "RelevanceLabel":
        pairs: set[tuple[str, int]] = set()
        doc_id = q.get("doc_id")
        if doc_id:
            pairs.update((doc_id, p) for p in q.get("relevant_pages", []))
        # Cross-document questions list additional documents that also answer.
        for extra_doc, pages in (q.get("extra_relevant") or {}).items():
            pairs.update((extra_doc, p) for p in pages)
        return cls(frozenset(pairs))

    def __bool__(self) -> bool:
        return bool(self.pairs)


def is_relevant(doc: Document, label: RelevanceLabel) -> bool:
    key = (doc.metadata.get("doc_id"), doc.metadata.get("page"))
    return key in label.pairs


def relevance_vector(docs: Sequence[Document], label: RelevanceLabel) -> list[bool]:
    return [is_relevant(d, label) for d in docs]


def recall_at_k(hits: Sequence[bool], label: RelevanceLabel, k: int) -> float:
    """Fraction of the question's relevant pages recovered in the top k.

    Denominator is the number of distinct relevant *pages*, not chunks, so a
    config that returns three chunks from the one relevant page does not score
    above 1.0 and does not get credit for redundancy.
    """
    if not label:
        return 0.0
    return min(1.0, sum(hits[:k]) / len(label.pairs)) if any(hits[:k]) else 0.0


def hit_at_k(hits: Sequence[bool], k: int) -> float:
    """1.0 if any relevant page appears in the top k.

    Reported alongside recall because for a single-page question the two are
    identical, and most questions here are single-page. Divergence between the
    two columns is the signal that multi-page questions are being handled badly.
    """
    return 1.0 if any(hits[:k]) else 0.0


def reciprocal_rank(hits: Sequence[bool]) -> float:
    """1/rank of the first relevant chunk, 0 if none.

    MRR is the metric that actually tracks reranking. Recall@k cannot see a
    reordering within the top k - it is rank-insensitive by construction - so a
    cross-encoder that moves the right passage from position 4 to position 1
    shows up here and nowhere else.
    """
    for i, h in enumerate(hits):
        if h:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(hits: Sequence[bool], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(hits[:k]) / k


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def aggregate(per_question: Sequence[dict[str, Any]], k: int) -> dict[str, float]:
    """Aggregate per-question rows into headline numbers.

    Retrieval metrics are averaged over answerable questions only. Unanswerable
    questions have no relevant page by construction, so including them would
    drag every retrieval metric toward zero by a constant that says nothing
    about retrieval quality. They are scored separately by abstention rate.
    """
    answerable = [r for r in per_question if r.get("answerable", True)]
    unanswerable = [r for r in per_question if not r.get("answerable", True)]

    out: dict[str, float] = {
        "n_questions": len(per_question),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        f"recall@{k}": mean(r["recall"] for r in answerable),
        f"hit@{k}": mean(r["hit"] for r in answerable),
        "mrr": mean(r["reciprocal_rank"] for r in answerable),
        f"precision@{k}": mean(r["precision"] for r in answerable),
    }

    judged = [r for r in per_question if r.get("judge_score") is not None]
    if judged:
        out["answer_correctness"] = mean(r["judge_score"] for r in judged)
        ans_judged = [r for r in judged if r.get("answerable", True)]
        if ans_judged:
            out["answer_correctness_answerable"] = mean(
                r["judge_score"] for r in ans_judged
            )

    # Abstention only means something when an answer was actually generated. In
    # retrieval-only mode no row has an `abstained` field, and reporting 0.0
    # would read as "it never abstained" rather than "not measured".
    generated = [r for r in per_question if "abstained" in r]
    if generated:
        gen_unanswerable = [r for r in generated if not r.get("answerable", True)]
        gen_answerable = [r for r in generated if r.get("answerable", True)]
        if gen_unanswerable:
            # Did the pipeline correctly refuse? This is the hallucination guard.
            out["abstention_accuracy"] = mean(
                1.0 if r.get("abstained") else 0.0 for r in gen_unanswerable
            )
        if gen_answerable:
            # How often did it wrongly refuse a question it had evidence for?
            out["false_abstention_rate"] = mean(
                1.0 if r.get("abstained") else 0.0 for r in gen_answerable
            )

    keyword = [r for r in per_question if r.get("keyword_score") is not None]
    if keyword:
        out["keyword_score"] = mean(r["keyword_score"] for r in keyword)

    out["mean_latency_s"] = mean(r.get("latency_seconds", 0.0) for r in per_question)
    return out


def by_type(per_question: Sequence[dict[str, Any]], k: int) -> dict[str, dict]:
    """Slice results by question type.

    The headline average hides the interesting part. Hybrid search is supposed
    to help lexical questions specifically; if the mean moves but the lexical
    slice does not, the improvement came from somewhere else and the stated
    reasoning is wrong.
    """
    types = sorted({r.get("type", "unknown") for r in per_question})
    return {
        t: aggregate([r for r in per_question if r.get("type") == t], k)
        for t in types
    }
