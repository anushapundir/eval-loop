"""Deterministic checks — free, structural evaluation (CLAUDE.md §7).

These run on the *full* dataset (no sampling) because they cost nothing. Each
check returns a :class:`CriterionScore` in [0, 1]. Day 1 ships the cheapest
structural checks; grounding and key-point coverage are added on Day 3.
"""

from __future__ import annotations

from agents.tools import tokenize
from storage.models import CriterionScore

# Reasonable answer-length window, in characters. Outside it, score degrades.
_MIN_CHARS = 20
_MAX_CHARS = 4000


def non_empty_check(text: str) -> CriterionScore:
    """1.0 if the response has visible content, else 0.0."""
    ok = bool(text and text.strip())
    return CriterionScore(
        name="non_empty",
        score=1.0 if ok else 0.0,
        justification=None if ok else "Response was empty.",
    )


def length_check(text: str) -> CriterionScore:
    """Score how well the response length sits within a sensible window."""
    n = len(text.strip())
    if n < _MIN_CHARS:
        score, why = n / _MIN_CHARS, f"Too short ({n} chars < {_MIN_CHARS})."
    elif n > _MAX_CHARS:
        score, why = max(0.0, _MAX_CHARS / n), f"Too long ({n} chars > {_MAX_CHARS})."
    else:
        score, why = 1.0, None
    return CriterionScore(name="length", score=round(score, 3), justification=why)


def grounding_check(text: str, context: str) -> CriterionScore:
    """Fraction of the response's content words that appear in the retrieved context.

    A grounding proxy (token overlap, not semantics): a high score means most of
    the response is drawn from the context rather than invented, which is what we
    want from a RAG answer. When *no* context was retrieved, grounding cannot be
    measured, so we return 1.0 and say so rather than penalize an honest "I don't
    know" — the length and judge criteria still constrain such answers.
    """
    response_tokens = tokenize(text)
    context_tokens = tokenize(context)
    if not context_tokens:
        return CriterionScore(
            name="grounding",
            score=1.0,
            justification="No retrieved context; grounding not applicable.",
        )
    if not response_tokens:
        return CriterionScore(name="grounding", score=0.0, justification="Empty response.")
    overlap = len(response_tokens & context_tokens) / len(response_tokens)
    why = None if overlap >= 0.5 else "Much of the response is not supported by the context."
    return CriterionScore(name="grounding", score=round(overlap, 3), justification=why)


def coverage_check(text: str, key_points: list[str]) -> CriterionScore:
    """Fraction of required key points present in the response.

    A key point counts as covered when all of its content tokens appear in the
    response (token-subset match, so "retrieved context" matches regardless of
    word order). With no key points there is nothing to cover, so the score is a
    perfect-but-uninformative 1.0.
    """
    if not key_points:
        return CriterionScore(
            name="coverage", score=1.0, justification="No key points to cover."
        )
    response_tokens = tokenize(text)
    covered = sum(
        1 for kp in key_points if tokenize(kp) and tokenize(kp) <= response_tokens
    )
    score = covered / len(key_points)
    why = None if covered == len(key_points) else f"Covered {covered}/{len(key_points)} key points."
    return CriterionScore(name="coverage", score=round(score, 3), justification=why)


def run_basic_checks(text: str) -> list[CriterionScore]:
    """Run the Day 1 deterministic checks over a response (structural only)."""
    return [non_empty_check(text), length_check(text)]


def run_deterministic_checks(
    text: str, *, context: str = "", key_points: list[str] | None = None
) -> list[CriterionScore]:
    """Run the full Day 3 deterministic suite over a response.

    Composes the structural checks (``non_empty``, ``length``) with grounding
    (against the retrieved ``context``) and key-point ``coverage``. All are free
    and run on the full dataset; only the LLM judge is sampled.
    """
    return [
        non_empty_check(text),
        length_check(text),
        grounding_check(text, context),
        coverage_check(text, key_points or []),
    ]
