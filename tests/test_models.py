"""Unit tests for the Pydantic data models (the integration boundary)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from storage.models import (
    AgentResponse,
    CriterionScore,
    EvalResult,
    Experiment,
    ResponseVersion,
    Task,
)


def test_task_defaults() -> None:
    task = Task(prompt="What is X?")
    assert task.id
    assert task.source == "user"
    assert task.key_points == []
    assert task.created_at is not None


def test_response_defaults_to_v1() -> None:
    resp = AgentResponse(task_id="t1", text="hello")
    assert resp.version is ResponseVersion.V1
    assert resp.model_provider == "ollama"


def test_criterion_score_bounds() -> None:
    CriterionScore(name="ok", score=0.0)
    CriterionScore(name="ok", score=1.0)
    with pytest.raises(ValidationError):
        CriterionScore(name="bad", score=1.5)
    with pytest.raises(ValidationError):
        CriterionScore(name="bad", score=-0.1)


def test_eval_result_overall_bounds() -> None:
    result = EvalResult(
        task_id="t1",
        response_id="r1",
        version=ResponseVersion.V1,
        deterministic=[CriterionScore(name="length", score=1.0)],
        overall_score=0.9,
        passed=True,
    )
    assert result.judged is False
    with pytest.raises(ValidationError):
        EvalResult(
            task_id="t1",
            response_id="r1",
            version=ResponseVersion.V1,
            overall_score=2.0,
        )


def test_experiment_optional_metrics() -> None:
    exp = Experiment(name="baseline")
    assert exp.mean_v1 is None
    assert exp.improvement_delta is None
    assert exp.prompt_version == "v1"
