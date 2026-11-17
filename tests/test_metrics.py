"""Tests for retrieval metrics.

These matter more than they look. Every claim in the results table is produced
by this module, so a silent bug here would not crash anything - it would just
make the numbers wrong in a way that survives into a README and an interview.
The rank-sensitivity tests in particular pin down the property that justifies
reporting MRR alongside recall.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from eval.metrics import (
    RelevanceLabel,
    aggregate,
    bootstrap_ci,
    hit_at_k,
    is_relevant,
    paired_bootstrap_ci,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    relevance_vector,
)


class TestRelevanceLabel:
    def test_builds_doc_page_pairs(self):
        label = RelevanceLabel.from_question(
            {"doc_id": "attention", "relevant_pages": [1, 8]}
        )
        assert label.pairs == {("attention", 1), ("attention", 8)}

    def test_includes_cross_document_pages(self):
        label = RelevanceLabel.from_question(
            {
                "doc_id": "bert",
                "relevant_pages": [3],
                "extra_relevant": {"attention": [3]},
            }
        )
        assert label.pairs == {("bert", 3), ("attention", 3)}

    def test_unanswerable_question_has_no_relevant_pages(self):
        label = RelevanceLabel.from_question(
            {"doc_id": None, "relevant_pages": [], "answerable": False}
        )
        assert not label

    def test_same_page_number_different_docs_is_not_relevant(self):
        """Guards the most likely labelling bug: matching on page alone."""
        label = RelevanceLabel.from_question(
            {"doc_id": "attention", "relevant_pages": [3]}
        )
        wrong_doc = Document(page_content="x", metadata={"doc_id": "bert", "page": 3})
        assert not is_relevant(wrong_doc, label)


class TestRecall:
    def test_full_recall_when_all_pages_found(self):
        label = RelevanceLabel.from_question(
            {"doc_id": "attention", "relevant_pages": [1, 8]}
        )
        assert recall_at_k([True, True, False, False], label, 4) == 1.0

    def test_partial_recall(self):
        label = RelevanceLabel.from_question(
            {"doc_id": "attention", "relevant_pages": [1, 8]}
        )
        assert recall_at_k([True, False, False, False], label, 4) == 0.5

    def test_zero_when_nothing_relevant(self):
        label = RelevanceLabel.from_question(
            {"doc_id": "attention", "relevant_pages": [1]}
        )
        assert recall_at_k([False] * 4, label, 4) == 0.0

    def test_k_truncates(self):
        """A hit below the cutoff must not count."""
        label = RelevanceLabel.from_question(
            {"doc_id": "attention", "relevant_pages": [1]}
        )
        hits = [False, False, False, True]
        assert recall_at_k(hits, label, 2) == 0.0
        assert recall_at_k(hits, label, 4) == 1.0

    def test_never_exceeds_one_with_duplicate_hits(self):
        """Two chunks from the same relevant page is not 200% recall."""
        label = RelevanceLabel.from_question(
            {"doc_id": "attention", "relevant_pages": [3]}
        )
        assert recall_at_k([True, True, False, False], label, 4) == 1.0

    def test_unanswerable_scores_zero_not_error(self):
        label = RelevanceLabel.from_question({"doc_id": None, "relevant_pages": []})
        assert recall_at_k([False] * 4, label, 4) == 0.0


class TestReciprocalRank:
    @pytest.mark.parametrize(
        "hits,expected",
        [
            ([True, False, False, False], 1.0),
            ([False, True, False, False], 0.5),
            ([False, False, True, False], 1 / 3),
            ([False, False, False, True], 0.25),
            ([False, False, False, False], 0.0),
        ],
    )
    def test_reciprocal_of_first_hit_rank(self, hits, expected):
        assert reciprocal_rank(hits) == pytest.approx(expected)

    def test_is_rank_sensitive_where_recall_is_not(self):
        """The property that makes MRR worth reporting.

        Reranking reorders within the top k. Recall@k cannot see that; MRR can.
        Without this distinction, a cross-encoder that promotes the right
        passage from rank 4 to rank 1 would appear to do nothing.
        """
        label = RelevanceLabel.from_question(
            {"doc_id": "attention", "relevant_pages": [3]}
        )
        before = [False, False, False, True]
        after = [True, False, False, False]
        assert recall_at_k(before, label, 4) == recall_at_k(after, label, 4)
        assert reciprocal_rank(after) > reciprocal_rank(before)


class TestHitAndPrecision:
    def test_hit_is_binary(self):
        assert hit_at_k([False, True, False], 3) == 1.0
        assert hit_at_k([False, False, False], 3) == 0.0

    def test_hit_respects_k(self):
        assert hit_at_k([False, False, True], 2) == 0.0

    def test_precision_counts_all_hits(self):
        assert precision_at_k([True, True, False, False], 4) == 0.5

    def test_precision_of_zero_k_is_zero_not_crash(self):
        assert precision_at_k([True], 0) == 0.0


class TestRelevanceVector:
    def test_marks_correct_position(self, retrieved_docs):
        label = RelevanceLabel.from_question(
            {"doc_id": "attention", "relevant_pages": [3]}
        )
        assert relevance_vector(retrieved_docs, label) == [False, False, True, False]


class TestAggregate:
    def _rows(self):
        return [
            {
                "answerable": True,
                "recall": 1.0,
                "hit": 1.0,
                "reciprocal_rank": 1.0,
                "precision": 0.25,
                "type": "factoid",
                "latency_seconds": 1.0,
                "judge_score": 1.0,
                "abstained": False,
            },
            {
                "answerable": True,
                "recall": 0.0,
                "hit": 0.0,
                "reciprocal_rank": 0.0,
                "precision": 0.0,
                "type": "factoid",
                "latency_seconds": 3.0,
                "judge_score": 0.0,
                "abstained": False,
            },
            {
                "answerable": False,
                "recall": 0.0,
                "hit": 0.0,
                "reciprocal_rank": 0.0,
                "precision": 0.0,
                "type": "unanswerable",
                "latency_seconds": 2.0,
                "judge_score": 1.0,
                "abstained": True,
            },
        ]

    def test_retrieval_metrics_exclude_unanswerable(self):
        """Unanswerable questions have no relevant page by construction.

        Including them would drag recall down by a constant that says nothing
        about retrieval quality.
        """
        out = aggregate(self._rows(), k=4)
        assert out["n_answerable"] == 2
        assert out["n_unanswerable"] == 1
        assert out["recall@4"] == 0.5  # 1.0 and 0.0 over 2 answerable, not 3

    def test_abstention_accuracy_uses_only_unanswerable(self):
        out = aggregate(self._rows(), k=4)
        assert out["abstention_accuracy"] == 1.0
        assert out["false_abstention_rate"] == 0.0

    def test_abstention_omitted_when_no_generation(self):
        """Retrieval-only rows have no `abstained` field.

        Reporting 0.0 there would read as 'never abstained' rather than
        'not measured', which is exactly the kind of number that ends up
        misquoted.
        """
        rows = [
            {
                "answerable": False,
                "recall": 0.0,
                "hit": 0.0,
                "reciprocal_rank": 0.0,
                "precision": 0.0,
                "type": "unanswerable",
            }
        ]
        out = aggregate(rows, k=4)
        assert "abstention_accuracy" not in out

    def test_answer_correctness_spans_all_judged_questions(self):
        out = aggregate(self._rows(), k=4)
        assert out["answer_correctness"] == pytest.approx(2 / 3, abs=1e-4)
        assert out["answer_correctness_answerable"] == 0.5

    def test_empty_input_does_not_crash(self):
        out = aggregate([], k=4)
        assert out["n_questions"] == 0


class TestBootstrap:
    def test_constant_values_give_zero_width_interval(self):
        assert bootstrap_ci([0.5] * 20) == (0.5, 0.5)

    def test_interval_contains_the_mean(self):
        vals = [1.0] * 10 + [0.0] * 10
        lo, hi = bootstrap_ci(vals)
        assert lo < 0.5 < hi

    def test_seeded_and_reproducible(self):
        vals = [0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 0.25]
        assert bootstrap_ci(vals) == bootstrap_ci(vals)

    def test_paired_identical_runs_have_zero_delta(self):
        """Pairing is the point: same values in both runs means no difference,
        however noisy the values are across questions."""
        vals = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]
        assert paired_bootstrap_ci(vals, vals) == (0.0, 0.0)

    def test_paired_consistent_gain_excludes_zero(self):
        a = [0.0, 0.2, 0.5, 0.1, 0.7, 0.3] * 4
        b = [x + 0.2 for x in a]
        lo, hi = paired_bootstrap_ci(a, b)
        assert lo > 0

    def test_paired_rejects_unequal_lengths(self):
        with pytest.raises(ValueError):
            paired_bootstrap_ci([1.0], [1.0, 0.0])
