"""Tests for metrics aggregation (reports/metrics.py).

Pure functions over the storage models — no DB, no model calls. Fixtures build
EvalResult/Experiment objects in memory with known scores so every aggregate is
hand-checkable.
"""

from __future__ import annotations

from datetime import UTC, datetime

from reports.metrics import (
    check_regression,
    compare_versions,
    experiment_trend,
    per_criterion_means,
    summarize_experiment,
)
from storage.models import (
    CriterionScore,
    EvalResult,
    Experiment,
    ResponseVersion,
)


def _result(
    version: ResponseVersion,
    overall: float,
    *,
    passed: bool,
    deterministic: list[tuple[str, float]] | None = None,
    judge: list[tuple[str, float]] | None = None,
) -> EvalResult:
    return EvalResult(
        task_id="t",
        response_id="r",
        version=version,
        deterministic=[CriterionScore(name=n, score=s) for n, s in (deterministic or [])],
        judge=[CriterionScore(name=n, score=s) for n, s in (judge or [])],
        overall_score=overall,
        passed=passed,
    )


# --- summarize_experiment ---------------------------------------------------


def test_summarize_experiment_means_and_pass_rates() -> None:
    exp = Experiment(name="loop-golden", n_tasks=2, n_judged=0)
    results = [
        _result(ResponseVersion.V1, 0.6, passed=False),
        _result(ResponseVersion.V1, 0.8, passed=True),
        _result(ResponseVersion.V2, 0.8, passed=True),
        _result(ResponseVersion.V2, 0.9, passed=True),
    ]
    summary = summarize_experiment(exp, results)

    assert summary.mean_v1 == 0.7  # (0.6 + 0.8) / 2
    assert summary.mean_v2 == 0.85  # (0.8 + 0.9) / 2
    assert summary.improvement_delta == 0.15
    assert summary.v1_pass_rate == 0.5  # 1 of 2
    assert summary.v2_pass_rate == 1.0


def test_summarize_experiment_v1_only_has_no_delta() -> None:
    exp = Experiment(name="eval-golden", n_tasks=2, n_judged=0)
    results = [
        _result(ResponseVersion.V1, 0.6, passed=False),
        _result(ResponseVersion.V1, 0.8, passed=True),
    ]
    summary = summarize_experiment(exp, results)

    assert summary.mean_v1 == 0.7
    assert summary.mean_v2 is None
    assert summary.improvement_delta is None
    assert summary.v2_pass_rate is None


# --- per_criterion_means ----------------------------------------------------


def test_per_criterion_means_splits_by_version() -> None:
    results = [
        _result(
            ResponseVersion.V1, 0.5, passed=False,
            deterministic=[("grounding", 0.4), ("coverage", 0.6)],
        ),
        _result(
            ResponseVersion.V2, 0.8, passed=True,
            deterministic=[("grounding", 0.8), ("coverage", 0.8)],
        ),
    ]
    table = per_criterion_means(results)

    assert table.loc["grounding", "v1"] == 0.4
    assert table.loc["grounding", "v2"] == 0.8
    assert table.loc["coverage", "v1"] == 0.6


def test_per_criterion_means_handles_missing_judge() -> None:
    """A no-judge run contributes only deterministic criteria; no crash."""
    results = [
        _result(
            ResponseVersion.V1, 0.5, passed=False,
            deterministic=[("grounding", 0.5)],
            judge=[],
        )
    ]
    table = per_criterion_means(results)

    assert list(table.index) == ["grounding"]
    assert "correctness" not in table.index


def test_per_criterion_means_empty_is_empty_frame() -> None:
    assert per_criterion_means([]).empty


# --- compare_versions -------------------------------------------------------


def test_compare_versions_computes_overall_and_per_criterion_delta() -> None:
    results = [
        _result(
            ResponseVersion.V1, 0.5, passed=False,
            deterministic=[("grounding", 0.4)],
        ),
        _result(
            ResponseVersion.V2, 0.8, passed=True,
            deterministic=[("grounding", 0.8)],
        ),
    ]
    cmp = compare_versions(results)

    assert cmp.overall_delta == 0.3  # 0.8 - 0.5
    assert cmp.per_criterion_delta == {"grounding": 0.4}


def test_compare_versions_v1_only_has_no_delta() -> None:
    results = [_result(ResponseVersion.V1, 0.5, passed=False, deterministic=[("g", 0.5)])]
    cmp = compare_versions(results)

    assert cmp.overall_delta is None
    assert cmp.per_criterion_delta == {}


# --- experiment_trend -------------------------------------------------------


def test_experiment_trend_sorted_chronologically() -> None:
    newer = Experiment(
        name="b", mean_v1=0.8, created_at=datetime(2026, 1, 2, tzinfo=UTC)
    )
    older = Experiment(
        name="a", mean_v1=0.7, created_at=datetime(2026, 1, 1, tzinfo=UTC)
    )
    # list_experiments returns newest-first; trend should re-sort oldest-first.
    trend = experiment_trend([newer, older])

    assert list(trend["name"]) == ["a", "b"]
    assert list(trend["mean_v1"]) == [0.7, 0.8]


def test_experiment_trend_empty_is_empty_frame() -> None:
    assert experiment_trend([]).empty


# --- check_regression -------------------------------------------------------


def test_check_regression_passes_on_improvement() -> None:
    res = check_regression(current_mean=0.88, baseline_mean=0.85)
    assert res.regressed is False
    assert res.delta == 0.03


def test_check_regression_flags_drop_beyond_tolerance() -> None:
    res = check_regression(current_mean=0.80, baseline_mean=0.85, tolerance=0.02)
    assert res.regressed is True
    assert res.delta == -0.05


def test_check_regression_within_tolerance_passes() -> None:
    res = check_regression(current_mean=0.84, baseline_mean=0.85, tolerance=0.02)
    assert res.regressed is False


def test_check_regression_no_baseline_passes() -> None:
    res = check_regression(current_mean=0.85, baseline_mean=None)
    assert res.regressed is False
    assert res.baseline_mean is None
