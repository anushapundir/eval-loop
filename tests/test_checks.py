"""Tests for deterministic checks (evaluators/checks.py).

All checks are pure functions returning a ``CriterionScore`` in [0, 1]. They run
on the full dataset (free), so they are tested directly with no model calls.
"""

from __future__ import annotations

from evaluators.checks import (
    coverage_check,
    grounding_check,
    run_deterministic_checks,
)

_CONTEXT = (
    "[rag] Grounding means the answer is supported by the retrieved context "
    "rather than invented, so it can be traced back to a source passage."
)


def test_grounding_high_when_response_uses_context() -> None:
    grounded = "The answer is supported by the retrieved context and traced to a source passage."
    score = grounding_check(grounded, _CONTEXT)
    assert score.name == "grounding"
    assert score.score >= 0.7


def test_grounding_low_when_response_is_off_topic() -> None:
    off_topic = "Bananas grow in tropical climates and ripen quickly after picking."
    score = grounding_check(off_topic, _CONTEXT)
    assert score.score < 0.4


def test_grounding_not_applicable_when_no_context() -> None:
    """With nothing retrieved, grounding cannot be measured and must not penalize."""
    score = grounding_check("I do not have enough information to answer.", "")
    assert score.score == 1.0
    assert score.justification is not None


def test_grounding_zero_for_empty_response() -> None:
    score = grounding_check("", _CONTEXT)
    assert score.score == 0.0


def test_coverage_fraction_of_key_points_present() -> None:
    text = "A retriever finds passages and the generator answers using them."
    score = coverage_check(text, ["retriever", "generator", "vector store"])
    assert score.name == "coverage"
    # 2 of 3 key points present.
    assert score.score == round(2 / 3, 3)


def test_coverage_full_when_no_key_points() -> None:
    """No key points means nothing to cover -> a perfect, uninformative 1.0."""
    score = coverage_check("anything at all", [])
    assert score.score == 1.0


def test_coverage_matches_multiword_key_point() -> None:
    text = "The answer is supported by the retrieved context here."
    score = coverage_check(text, ["supported by retrieved context"])
    assert score.score == 1.0


def test_run_deterministic_checks_composes_all_four() -> None:
    scores = run_deterministic_checks(
        "A grounded answer supported by the retrieved context and its source.",
        context=_CONTEXT,
        key_points=["retrieved context"],
    )
    names = [s.name for s in scores]
    assert names == ["non_empty", "length", "grounding", "coverage"]
    assert all(0.0 <= s.score <= 1.0 for s in scores)
