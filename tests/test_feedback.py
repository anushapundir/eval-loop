"""Tests for the deterministic feedback generator (feedback/generator.py).

The generator is a pure function: an EvalResult + the task + the response text in,
a structured Feedback out. No model calls, so these run fully offline. It turns
low deterministic scores into actionable critique — most importantly naming the
specific key points an answer is missing.
"""

from __future__ import annotations

from config.settings import Settings
from feedback.generator import generate_feedback
from storage.models import CriterionScore, EvalResult, ResponseVersion, Task


def _settings() -> Settings:
    return Settings(pass_threshold=0.7)


def _eval(scores: list[CriterionScore]) -> EvalResult:
    overall = sum(s.score for s in scores) / len(scores)
    return EvalResult(
        task_id="t1",
        response_id="r1",
        version=ResponseVersion.V1,
        deterministic=scores,
        overall_score=round(overall, 3),
        passed=overall >= 0.7,
    )


def test_feedback_names_missing_key_points() -> None:
    """Low coverage produces an item naming the key points the answer missed."""
    task = Task(prompt="what is RAG", key_points=["retriever", "generator", "fetched"])
    result = _eval(
        [
            CriterionScore(name="non_empty", score=1.0),
            CriterionScore(name="length", score=1.0),
            CriterionScore(name="grounding", score=0.9),
            CriterionScore(name="coverage", score=0.333, justification="Covered 1/3 key points."),
        ]
    )
    # Response mentions only "retriever" — generator missing, fetched missing.
    feedback = generate_feedback(result, task, "a retriever finds passages", settings=_settings())

    assert feedback.is_actionable
    cov = [i for i in feedback.items if i.criterion == "coverage"]
    assert cov, "expected a coverage feedback item"
    assert "generator" in cov[0].suggestion
    assert "fetched" in cov[0].suggestion
    assert "retriever" not in cov[0].suggestion  # already covered, not re-listed
    assert "generator" in feedback.text


def test_feedback_flags_low_grounding() -> None:
    task = Task(prompt="q", key_points=[])
    result = _eval(
        [
            CriterionScore(name="non_empty", score=1.0),
            CriterionScore(name="length", score=1.0),
            CriterionScore(name="grounding", score=0.2, justification="Not supported."),
            CriterionScore(name="coverage", score=1.0),
        ]
    )
    feedback = generate_feedback(result, task, "an unsupported answer", settings=_settings())

    assert feedback.is_actionable
    assert any(i.criterion == "grounding" for i in feedback.items)


def test_feedback_empty_when_response_is_good() -> None:
    """A passing response with full coverage yields no actionable feedback."""
    task = Task(prompt="q", key_points=["alpha"])
    result = _eval(
        [
            CriterionScore(name="non_empty", score=1.0),
            CriterionScore(name="length", score=1.0),
            CriterionScore(name="grounding", score=0.9),
            CriterionScore(name="coverage", score=1.0),
        ]
    )
    feedback = generate_feedback(result, task, "alpha is the answer", settings=_settings())

    assert not feedback.is_actionable
    assert feedback.items == []
