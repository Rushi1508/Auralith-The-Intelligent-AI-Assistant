"""The RAG answer chain.

Differences from the inherited chain that matter:

* It returns the retrieved passages alongside the answer. The prototype piped
  the retriever straight into the prompt, so the passages existed only as an
  intermediate value and neither the UI nor an evaluator could see them. You
  cannot measure retrieval you cannot observe, so this is a prerequisite for
  Phase A, not a UI nicety.
* Passages are formatted with explicit source labels and the model is asked to
  cite them, which gives a cheap, checkable grounding signal.
* It is abstention-aware. The prototype's prompt said "answer based only on the
  following context" but gave the model no way out, so a missing answer
  produced a confident fabrication. Here, an explicit "context does not contain
  the answer" escape is offered - and the eval set includes questions where
  taking it is the correct behaviour.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from auralith.config import RetrievalConfig, Settings
from auralith.providers import get_chat_model
from auralith.retrievers import retrieve

log = logging.getLogger(__name__)

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are Auralith, a document question-answering assistant.

Answer the question using ONLY the numbered context passages below. Follow these rules:
- Cite the passage numbers you used, like [1] or [2][3].
- Be specific. If the answer is a number, a name, or a term, state it exactly.
- If the question has several parts, answer every part the passages support.
  If a part is not covered, say which part is missing - but still answer the rest.
- Only if the passages contain nothing that answers any part of the question,
  reply exactly:
  NOT_IN_CONTEXT
- Do not guess and do not use outside knowledge.

Context passages:
{context}

Question: {question}

Answer:"""
)

NOT_IN_CONTEXT = "NOT_IN_CONTEXT"


def is_abstention(text: str) -> bool:
    """True if the answer is a refusal, not a partial answer.

    The prompt allows answering part of a multi-part question and flagging the
    missing part, so the token can appear mid-answer. Only a reply that *leads*
    with it is a refusal. A substring check would score "Dec 2018 dump, 100-word
    chunks; the document count is NOT_IN_CONTEXT" as a refusal and throw away
    two correct facts.
    """
    return (text or "").lstrip(" *_`>#\n\t").upper().startswith(NOT_IN_CONTEXT)


@dataclass
class RagResult:
    """An answer plus everything needed to audit how it was produced."""

    question: str
    answer: str
    passages: list[Document]
    latency_seconds: float
    config_slug: str
    abstained: bool = False
    error: str | None = None
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def citations(self) -> list[dict[str, object]]:
        return [
            {
                "n": i + 1,
                "doc_id": d.metadata.get("doc_id"),
                "page": d.metadata.get("page"),
                "chunk_id": d.metadata.get("chunk_id"),
            }
            for i, d in enumerate(self.passages)
        ]


def format_passages(docs: list[Document]) -> str:
    """Render passages with numbered, citable source labels."""
    if not docs:
        return "(no passages retrieved)"
    return "\n\n".join(
        f"[{i + 1}] (source: {d.metadata.get('doc_id', '?')}, "
        f"page {d.metadata.get('page', '?')})\n{d.page_content.strip()}"
        for i, d in enumerate(docs)
    )


def answer_question(
    question: str,
    retriever: BaseRetriever,
    config: RetrievalConfig,
    settings: Settings,
) -> RagResult:
    """Retrieve, generate, and return both the answer and its evidence.

    Errors are captured into the result rather than raised. A sweep runs
    hundreds of questions; one malformed page or one model timeout should cost
    a single data point, not the whole run.
    """
    started = time.perf_counter()
    passages: list[Document] = []
    try:
        passages = retrieve(retriever, question, config)
        prompt = ANSWER_PROMPT.format_messages(
            context=format_passages(passages), question=question
        )
        raw = get_chat_model(settings).invoke(prompt)
        text = (raw.content if hasattr(raw, "content") else str(raw)).strip()
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
        log.warning("Failed on %r: %s", question[:60], exc)
        return RagResult(
            question=question,
            answer="",
            passages=passages,
            latency_seconds=round(time.perf_counter() - started, 3),
            config_slug=config.slug,
            error=f"{type(exc).__name__}: {exc}",
        )

    return RagResult(
        question=question,
        answer=text,
        passages=passages,
        latency_seconds=round(time.perf_counter() - started, 3),
        config_slug=config.slug,
        abstained=is_abstention(text),
    )
