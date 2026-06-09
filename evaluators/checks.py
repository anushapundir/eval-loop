"""Deterministic checks — free, structural evaluation (CLAUDE.md §7).

These run on the *full* dataset (no sampling) because they cost nothing. Each
check returns a :class:`CriterionScore` in [0, 1]. Day 1 ships the cheapest
structural checks; grounding and key-point coverage are added on Day 3.
"""

from __future__ import annotations

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


def run_basic_checks(text: str) -> list[CriterionScore]:
    """Run the Day 1 deterministic checks over a response."""
    return [non_empty_check(text), length_check(text)]
