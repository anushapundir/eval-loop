"""Tests for failure-mode clustering (evaluators/analysis.py).

Pure over the storage models; synthetic results with known weak criteria make
the mode counts hand-checkable. No DB, no model calls.
"""

from __future__ import annotations

from config.settings import Settings
from evaluators.analysis import cluster_failures
from storage.models import CriterionScore, EvalResult, ResponseVersion


def _settings(**overrides) -> Settings:
    base = dict(pass_threshold=0.7)
    base.update(overrides)
    return Settings(**base)


def _result(
    overall: float,
    *,
    passed: bool,
    deterministic: list[tuple[str, float]] | None = None,
    judge: list[tuple[str, float]] | None = None,
) -> EvalResult:
    return EvalResult(
        task_id="t",
        response_id="r",
        version=ResponseVersion.V1,
        deterministic=[CriterionScore(name=n, score=s) for n, s in (deterministic or [])],
        judge=[CriterionScore(name=n, score=s) for n, s in (judge or [])],
        overall_score=overall,
        passed=passed,
    )


def test_cluster_failures_attributes_weak_criteria_to_modes() -> None:
    results = [
        _result(0.5, passed=False, deterministic=[("grounding", 0.3), ("coverage", 0.4)]),
        _result(0.6, passed=False, deterministic=[("grounding", 0.5), ("length", 1.0)]),
    ]
    analysis = cluster_failures(results, settings=_settings())

    assert analysis.n_total == 2
    assert analysis.n_failed == 2
    # grounding weak in both; coverage weak in one; length cleared in both.
    assert analysis.mode_counts["low_grounding"] == 2
    assert analysis.mode_counts["missing_coverage"] == 1
    assert "length_violation" not in analysis.mode_counts


def test_cluster_failures_ignores_passing_results() -> None:
    results = [
        _result(0.9, passed=True, deterministic=[("grounding", 0.9)]),
        _result(0.4, passed=False, deterministic=[("grounding", 0.2)]),
    ]
    analysis = cluster_failures(results, settings=_settings())

    assert analysis.n_failed == 1
    assert analysis.mode_counts == {"low_grounding": 1}


def test_cluster_failures_maps_judge_criteria() -> None:
    results = [_result(0.5, passed=False, judge=[("correctness", 0.3), ("clarity", 0.9)])]
    analysis = cluster_failures(results, settings=_settings())

    assert analysis.mode_counts == {"incorrect": 1}


def test_cluster_failures_other_when_no_weak_criterion() -> None:
    """A failure whose individual criteria all clear the cutoff lands in 'other'."""
    results = [_result(0.6, passed=False, deterministic=[("grounding", 0.8)])]
    analysis = cluster_failures(results, settings=_settings())

    assert analysis.mode_counts == {"other": 1}


def test_cluster_failures_all_passing_is_empty() -> None:
    results = [_result(0.9, passed=True, deterministic=[("grounding", 0.9)])]
    analysis = cluster_failures(results, settings=_settings())

    assert analysis.n_failed == 0
    assert analysis.mode_counts == {}
