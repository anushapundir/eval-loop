"""Tests for chart rendering (reports/charts.py).

Charts are pure renderers, so we assert the PNG is written and non-empty (pixel
content isn't asserted). All output goes to ``tmp_path``; the Agg backend means
no display is needed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from evaluators.analysis import FailureAnalysis
from reports.charts import (
    chart_failure_modes,
    chart_per_criterion,
    chart_score_trend,
    chart_v1_vs_v2,
    render_all,
)
from reports.metrics import ExperimentSummary, experiment_trend, per_criterion_means
from storage.models import CriterionScore, EvalResult, Experiment, ResponseVersion


def _nonempty(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 0


def _result(version: ResponseVersion, overall: float, *, passed: bool, crit: str, score: float):
    return EvalResult(
        task_id="t", response_id="r", version=version,
        deterministic=[CriterionScore(name=crit, score=score)],
        overall_score=overall, passed=passed,
    )


def test_chart_score_trend_writes_png(tmp_path: Path) -> None:
    exps = [
        Experiment(name="a", mean_v1=0.84, mean_v2=0.87,
                   created_at=datetime(2026, 1, 1, tzinfo=UTC)),
        Experiment(name="b", mean_v1=0.85, mean_v2=0.88,
                   created_at=datetime(2026, 1, 2, tzinfo=UTC)),
    ]
    path = chart_score_trend(experiment_trend(exps), tmp_path)
    assert _nonempty(path)


def test_chart_score_trend_empty_still_writes(tmp_path: Path) -> None:
    path = chart_score_trend(experiment_trend([]), tmp_path)
    assert _nonempty(path)


def test_chart_v1_vs_v2_writes_png(tmp_path: Path) -> None:
    summary = ExperimentSummary(
        experiment_id="e", name="loop-golden", n_tasks=6, n_judged=0,
        mean_v1=0.84, mean_v2=0.87, improvement_delta=0.03,
    )
    path = chart_v1_vs_v2(summary, tmp_path)
    assert _nonempty(path)


def test_chart_per_criterion_writes_png(tmp_path: Path) -> None:
    results = [
        _result(ResponseVersion.V1, 0.5, passed=False, crit="grounding", score=0.4),
        _result(ResponseVersion.V2, 0.8, passed=True, crit="grounding", score=0.8),
    ]
    path = chart_per_criterion(per_criterion_means(results), tmp_path)
    assert _nonempty(path)


def test_chart_failure_modes_writes_png(tmp_path: Path) -> None:
    analysis = FailureAnalysis(n_total=6, n_failed=2, mode_counts={"low_grounding": 2})
    path = chart_failure_modes(analysis, tmp_path)
    assert _nonempty(path)


def test_chart_failure_modes_empty_still_writes(tmp_path: Path) -> None:
    analysis = FailureAnalysis(n_total=6, n_failed=0, mode_counts={})
    path = chart_failure_modes(analysis, tmp_path)
    assert _nonempty(path)


def test_render_all_emits_four_charts(tmp_path: Path) -> None:
    exp = Experiment(id="e", name="loop-golden", n_tasks=2, n_judged=0)
    results = [
        _result(ResponseVersion.V1, 0.6, passed=False, crit="grounding", score=0.5),
        _result(ResponseVersion.V2, 0.85, passed=True, crit="grounding", score=0.9),
    ]
    paths = render_all(exp, results, [exp], out_dir=tmp_path)

    assert set(paths) == {"score_trend", "v1_vs_v2", "per_criterion", "failure_modes"}
    assert all(_nonempty(p) for p in paths.values())
