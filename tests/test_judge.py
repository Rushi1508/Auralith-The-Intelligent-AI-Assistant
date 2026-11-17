"""Tests for judge parsing and scoring.

The parsing tests exist because small models wrap JSON in prose or fences a
meaningful fraction of the time. Every unparsed verdict silently becomes a zero,
which would drag `answer_correctness` down for reasons that have nothing to do
with the pipeline being measured. A parser bug here looks exactly like a
regression in answer quality, which is the worst kind of bug to have in an
eval harness.
"""

from __future__ import annotations

import pytest

from auralith.rag import is_abstention
from eval.judge import _extract_json, judge_answer, keyword_score


class FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class FakeJudge:
    def __init__(self, *responses: str) -> None:
        self.responses = list(responses) or [""]
        self.calls = 0
        self.prompts: list[str] = []

    def invoke(self, prompt, **_: object) -> FakeResponse:
        self.prompts.append(prompt)
        idx = min(self.calls, len(self.responses) - 1)
        self.calls += 1
        return FakeResponse(self.responses[idx])


class ExplodingJudge:
    def invoke(self, prompt, **_: object):
        raise RuntimeError("judge backend down")


class TestExtractJson:
    def test_bare_object(self):
        assert _extract_json('{"correct": true}') == {"correct": True}

    def test_fenced_json(self):
        assert _extract_json('```json\n{"correct": false}\n```') == {"correct": False}

    def test_fenced_without_language_tag(self):
        assert _extract_json('```\n{"correct": true}\n```') == {"correct": True}

    def test_object_surrounded_by_prose(self):
        got = _extract_json('Sure thing! {"correct": true, "reason": "ok"} Hope that helps.')
        assert got["correct"] is True

    def test_returns_none_for_no_json(self):
        assert _extract_json("I think it is correct.") is None

    def test_returns_none_for_a_bare_list(self):
        assert _extract_json("[1, 2, 3]") is None

    def test_returns_none_for_empty(self):
        assert _extract_json("") is None


class TestJudgeAnswer:
    def test_scores_a_correct_verdict(self):
        v = judge_answer(FakeJudge('{"correct": true, "reason": "matches"}'), "q", "ref", "cand")
        assert v.score == 1.0
        assert not v.parse_failed

    def test_scores_an_incorrect_verdict(self):
        v = judge_answer(FakeJudge('{"correct": false, "reason": "wrong number"}'), "q", "ref", "cand")
        assert v.score == 0.0

    def test_accepts_string_booleans(self):
        """Small models emit "true" as a string often enough to matter."""
        v = judge_answer(FakeJudge('{"correct": "true"}'), "q", "ref", "cand")
        assert v.score == 1.0

    def test_retries_then_succeeds(self):
        judge = FakeJudge("no json at all", '{"correct": true}')
        v = judge_answer(judge, "q", "ref", "cand", retries=1)
        assert v.score == 1.0
        assert judge.calls == 2

    def test_flags_parse_failure_after_retries(self):
        """A parse failure must be visible, not silently scored as wrong."""
        v = judge_answer(FakeJudge("never json", "still not json"), "q", "ref", "cand", retries=1)
        assert v.parse_failed is True
        assert v.score == 0.0

    def test_backend_failure_does_not_raise(self):
        v = judge_answer(ExplodingJudge(), "q", "ref", "cand", retries=0)
        assert v.score == 0.0
        assert v.parse_failed is True

    def test_citation_markers_are_stripped_before_judging(self):
        judge = FakeJudge('{"correct": true}')
        judge_answer(judge, "q", "ref", "[2] (source: bert, page 3)\nN = 6 [1][3].")
        assert "CANDIDATE ANSWER: N = 6 ." in judge.prompts[0]


class TestAbstentionIsDeterministic:
    """Abstention is a string check, never a model call.

    It is the one part of scoring with no ambiguity, so putting a model in the
    loop would only add noise.
    """

    def test_correct_abstention_on_unanswerable(self):
        judge = FakeJudge("SHOULD NOT BE CALLED")
        v = judge_answer(judge, "q", "NOT_IN_CONTEXT", "NOT_IN_CONTEXT", answerable=False)
        assert v.score == 1.0
        assert judge.calls == 0

    def test_hallucination_on_unanswerable_scores_zero(self):
        judge = FakeJudge("SHOULD NOT BE CALLED")
        v = judge_answer(judge, "q", "NOT_IN_CONTEXT", "The answer is 29.1 BLEU.", answerable=False)
        assert v.score == 0.0
        assert judge.calls == 0

    def test_abstaining_on_an_answerable_question_scores_zero(self):
        judge = FakeJudge("SHOULD NOT BE CALLED")
        v = judge_answer(judge, "q", "Six layers.", "NOT_IN_CONTEXT", answerable=True)
        assert v.score == 0.0
        assert judge.calls == 0

    def test_empty_answer_scores_zero_without_calling_the_judge(self):
        judge = FakeJudge("SHOULD NOT BE CALLED")
        assert judge_answer(judge, "q", "ref", "   ").score == 0.0
        assert judge.calls == 0


class TestKeywordScore:
    def test_all_keys_present(self):
        assert keyword_score("The encoder has 6 layers, d_model 512.", ["6", "512"]) == 1.0

    def test_partial(self):
        assert keyword_score("The encoder has 6 layers.", ["6", "512"]) == 0.5

    def test_case_insensitive(self):
        assert keyword_score("uses WORDPIECE tokens", ["wordpiece"]) == 1.0

    def test_returns_none_when_no_keys_declared(self):
        """None so the question is excluded from the mean, not counted as zero."""
        assert keyword_score("anything", []) is None

    @pytest.mark.parametrize("answer", ["", None])
    def test_empty_answer_scores_zero(self, answer):
        assert keyword_score(answer, ["6"]) == 0.0


class TestPartialAnswersAreNotAbstentions:
    """The prompt lets the model answer part of a multi-part question and flag
    the missing part. Only a reply that leads with the token is a refusal."""

    @pytest.mark.parametrize(
        "text",
        ["NOT_IN_CONTEXT", "  not_in_context", "**NOT_IN_CONTEXT**", "\nNOT_IN_CONTEXT."],
    )
    def test_leading_token_is_abstention(self, text):
        assert is_abstention(text)

    def test_partial_answer_mentioning_token_is_not_abstention(self):
        assert not is_abstention(
            "Dec 2018 dump, 100-word chunks [2]. The document count is NOT_IN_CONTEXT."
        )

    def test_partial_answer_goes_to_the_judge(self):
        judge = FakeJudge('{"correct": true, "reason": "ok"}')
        v = judge_answer(
            judge, "q", "Adam, b1=0.9", "Adam [1]. Epsilon is NOT_IN_CONTEXT.", answerable=True
        )
        assert judge.calls == 1
        assert v.score == 1.0
