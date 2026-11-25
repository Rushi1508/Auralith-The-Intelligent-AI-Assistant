"""LLM-as-judge scoring for answer correctness.

Design choices, each of which an interviewer can reasonably push on:

1. **Binary verdict, not a 1-5 scale.** A 7B model is not reliable at
   distinguishing a 3 from a 4, and averaging unreliable ordinals produces a
   number that looks precise and is not. Binary correct/incorrect is a
   judgement it can actually make, and the mean of a binary is an accuracy with
   an honest interpretation.

2. **The judge never sees the retrieved passages.** It compares the prediction
   against the human-written gold answer only. If it saw the context it would
   partly be grading whether the answer follows from what was retrieved, which
   is a different question and would make a config that retrieves nothing and
   correctly abstains look "correct" on questions that had a real answer.

3. **Abstention is scored deterministically, not by the judge.** Whether the
   output leads with NOT_IN_CONTEXT is a string check. Handing that to a model
   would add noise to the one part of the pipeline that has no ambiguity.

4. **A deterministic keyword check runs alongside.** `answer_keys` in the eval
   set gives a model-free correctness proxy. It is too strict to be the headline
   metric (a correct answer can paraphrase), but tracking judge/keyword
   agreement across a sweep is how judge drift gets caught. If they diverge,
   trust neither until you have looked.

Judge reliability is measured, not assumed - see `validate_judge`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel

from auralith.rag import NOT_IN_CONTEXT, is_abstention

log = logging.getLogger(__name__)

# Reasoning-before-verdict and question-scoped facts. The first prompt asked
# for "all key facts in the reference", so a 7B judge failed answers for omitting
# background the question never asked for, and for true extra detail; it agreed
# with hand labels on only 15/24 real answers (see judge_labels.json).
JUDGE_PROMPT = """You are grading a question-answering system against a reference answer.

Step 1: List the facts the QUESTION asks for, taking their correct values from the REFERENCE ANSWER. Only facts the question asks for are required. Anything else in the reference is background and is NOT required.
Step 2: For each required fact, check whether the CANDIDATE ANSWER states it with the same value. Paraphrase and different wording are fine.
Step 3: The candidate is correct only if every required fact is present and nothing in it contradicts the reference or is clearly false. Extra detail that the reference simply does not mention is fine.
A candidate that says it cannot answer is incorrect.

QUESTION: {question}

REFERENCE ANSWER: {reference}

CANDIDATE ANSWER: {candidate}

Respond with ONLY a JSON object, no other text:
{{"required_facts": ["..."], "missing_or_wrong": ["..."], "correct": true or false}}"""

# Citation markers like "[2]" or "(source: bert, page 3)" carry no facts and a
# bare "[1]" line made the judge report the answer as missing content.
_CITATION = re.compile(r"\[\d+\](?:\s*\(source:[^)]*\))?")


@dataclass
class JudgeVerdict:
    score: float
    reason: str
    raw: str = ""
    parse_failed: bool = False


def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull a JSON object out of a model response.

    Small models wrap JSON in prose or fences roughly 10-20% of the time. The
    original prototype's StructuredOutputParser would simply raise here and lose
    the whole run, which is why this degrades instead.
    """
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def judge_answer(
    judge: BaseChatModel,
    question: str,
    reference: str,
    candidate: str,
    answerable: bool = True,
    retries: int = 1,
    strip_citations: bool = True,
) -> JudgeVerdict:
    """Score one answer. Never raises."""
    # Abstention: fully deterministic, no model needed.
    abstained = is_abstention(candidate)
    if not answerable:
        return JudgeVerdict(
            score=1.0 if abstained else 0.0,
            reason="Correctly abstained." if abstained else "Should have abstained.",
        )
    if not (candidate or "").strip():
        return JudgeVerdict(score=0.0, reason="Empty answer.")
    if abstained:
        return JudgeVerdict(score=0.0, reason="Abstained on an answerable question.")

    # Off for code answers: the pattern would turn `arr[0]` into `arr`.
    cleaned = _CITATION.sub("", candidate) if strip_citations else candidate
    prompt = JUDGE_PROMPT.format(
        question=question, reference=reference, candidate=cleaned.strip()
    )
    last_raw = ""
    for attempt in range(retries + 1):
        try:
            resp = judge.invoke(prompt)
            last_raw = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as exc:  # noqa: BLE001
            log.warning("Judge call failed (attempt %d): %s", attempt + 1, exc)
            continue

        parsed = _extract_json(last_raw)
        if parsed is not None and "correct" in parsed:
            verdict = parsed["correct"]
            if isinstance(verdict, str):
                verdict = verdict.strip().lower() in {"true", "yes", "correct"}
            return JudgeVerdict(
                score=1.0 if verdict else 0.0,
                reason=str(parsed.get("missing_or_wrong") or parsed.get("reason", ""))[:300],
                raw=last_raw[:500],
            )

    log.warning("Judge produced unparseable output for %r", question[:60])
    return JudgeVerdict(
        score=0.0,
        reason="Judge output could not be parsed.",
        raw=last_raw[:500],
        parse_failed=True,
    )


def keyword_score(candidate: str, answer_keys: list[str]) -> float | None:
    """Deterministic proxy: fraction of required key strings present.

    Case-insensitive substring match. Returns None when a question declares no
    keys, so it is excluded from the average rather than counted as zero.
    """
    if not answer_keys:
        return None
    hay = (candidate or "").lower()
    return round(sum(1 for key in answer_keys if key.lower() in hay) / len(answer_keys), 4)


def validate_judge(
    judge: BaseChatModel, cases: list[dict[str, Any]]
) -> dict[str, Any]:
    """Measure the judge against hand-labelled cases.

    Deliberately includes adversarial pairs - a right number with wrong units, a
    paraphrase with no shared vocabulary, a confidently wrong answer. Reporting
    this agreement figure is what separates "we used an LLM judge" from "we know
    how much the LLM judge can be trusted", and the gap between the two is most
    of what an interviewer is probing for.
    """
    rows = []
    for c in cases:
        v = judge_answer(
            judge,
            c["question"],
            c["reference"],
            c["candidate"],
            answerable=c.get("answerable", True),
        )
        rows.append(
            {
                "id": c.get("id"),
                "question": c["question"],
                "candidate": c["candidate"],
                "expected": c["expected"],
                "judged": v.score,
                "agree": v.score == c["expected"],
                "reason": v.reason,
            }
        )
    agree = sum(1 for r in rows if r["agree"])
    return {
        "n_cases": len(rows),
        "n_agree": agree,
        "agreement": round(agree / len(rows), 4) if rows else 0.0,
        "disagreements": [r for r in rows if not r["agree"]],
        "cases": rows,
    }


# Hand-labelled cases. `expected` is the human verdict this file asserts is
# correct; the judge is measured against it.
JUDGE_VALIDATION_CASES: list[dict[str, Any]] = [
    {
        "question": "How many identical layers are stacked in the Transformer encoder?",
        "reference": "Six. The encoder is composed of a stack of N = 6 identical layers.",
        "candidate": "The encoder has 6 identical layers.",
        "expected": 1.0,
    },
    {
        "question": "How many identical layers are stacked in the Transformer encoder?",
        "reference": "Six. The encoder is composed of a stack of N = 6 identical layers.",
        "candidate": "The encoder is composed of 12 identical layers.",
        "expected": 0.0,
    },
    {
        "question": "How many identical layers are stacked in the Transformer encoder?",
        "reference": "Six. The encoder is composed of a stack of N = 6 identical layers.",
        "candidate": "N = 6.",
        "expected": 1.0,
    },
    {
        "question": "What percentage of WordPiece tokens does BERT mask?",
        "reference": "15% of all WordPiece tokens in each sequence, chosen at random.",
        "candidate": "BERT masks about one in seven tokens at random.",
        "expected": 1.0,
        "note": "Paraphrase with no shared numeral. Tests fact-vs-wording grading.",
    },
    {
        "question": "What percentage of WordPiece tokens does BERT mask?",
        "reference": "15% of all WordPiece tokens in each sequence, chosen at random.",
        "candidate": "15 tokens per sequence are masked.",
        "expected": 0.0,
        "note": "Right numeral, wrong unit. Tests that the judge is not keyword matching.",
    },
    {
        "question": "How many parameters does BART-large have?",
        "reference": "400M parameters.",
        "candidate": "BART-large has 400 million parameters, and was pre-trained with a "
        "denoising objective.",
        "expected": 1.0,
        "note": "Correct plus extra true detail.",
    },
    {
        "question": "How many parameters does BART-large have?",
        "reference": "400M parameters.",
        "candidate": "BART-large has 400 million parameters and 24 encoder layers "
        "trained on the December 2018 Wikipedia dump.",
        "expected": 0.0,
        "note": "Correct core fact contaminated with a fabricated detail.",
    },
    {
        "question": "What BLEU score did the Transformer achieve on WMT 2014 "
        "English-to-Romanian?",
        "reference": "NOT_IN_CONTEXT",
        "candidate": "NOT_IN_CONTEXT",
        "expected": 1.0,
        "answerable": False,
    },
    {
        "question": "What BLEU score did the Transformer achieve on WMT 2014 "
        "English-to-Romanian?",
        "reference": "NOT_IN_CONTEXT",
        "candidate": "The Transformer achieved 29.1 BLEU on English-to-Romanian.",
        "expected": 0.0,
        "answerable": False,
        "note": "The exact hallucination the unanswerable questions exist to catch.",
    },
    {
        "question": "What optimizer did the Transformer authors use, and what beta "
        "and epsilon values?",
        "reference": "Adam, with beta_1 = 0.9, beta_2 = 0.98 and epsilon = 10^-9.",
        "candidate": "They used the Adam optimizer.",
        "expected": 0.0,
        "note": "Partially correct: omits required beta values. Tests the "
        "'all key facts must be present' rule.",
    },
]


# Real answers from a full run (results/production_partial.json), hand-labelled.
# The synthetic cases above test edge cases; these test the answers the judge
# actually sees, which is where the 7B judge turned out to fail.
JUDGE_LABELS_PATH = Path(__file__).resolve().parent / "judge_labels.json"


def load_judge_labels(path: Path = JUDGE_LABELS_PATH) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))
