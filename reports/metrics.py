"""Metrics aggregation — pandas over stored eval results (CLAUDE.md §4/§8).

This module is the *numbers* half of the reporting layer: it turns lists of
``EvalResult`` / ``Experiment`` records (read elsewhere through ``storage/db.py``)
into aggregated metrics — per-criterion means, overall scores, v1-vs-v2 deltas,
pass-rates, an across-experiments trend, and a regression verdict.

It is deliberately *pure over the storage models*: every function takes already
loaded records and returns typed results (Pydantic models for scalars, pandas
DataFrames for the chart-feeding tables). It never touches SQLite or renders a
chart, so the CLI orchestrates the reads and ``charts.py`` consumes the output.
"""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel

from storage.models import EvalResult, Experiment, ResponseVersion


class ExperimentSummary(BaseModel):
    """Aggregated headline metrics for a single experiment."""

    experiment_id: str
    name: str
    n_tasks: int
    n_judged: int
    mean_v1: float | None = None
    mean_v2: float | None = None
    improvement_delta: float | None = None
    v1_pass_rate: float | None = None
    v2_pass_rate: float | None = None


class VersionComparison(BaseModel):
    """v1-vs-v2 comparison: overall delta plus a per-criterion breakdown."""

    mean_v1: float | None = None
    mean_v2: float | None = None
    overall_delta: float | None = None
    per_criterion_delta: dict[str, float] = {}


class RegressionResult(BaseModel):
    """Verdict of comparing a run's mean against a baseline (the prior run)."""

    regressed: bool
    current_mean: float
    baseline_mean: float | None = None
    delta: float | None = None
    tolerance: float = 0.0
    message: str = ""


def _by_version(results: list[EvalResult], version: ResponseVersion) -> list[EvalResult]:
    """Filter eval results to one response version (v1 or v2)."""
    return [r for r in results if r.version is version]


def _mean(results: list[EvalResult]) -> float | None:
    """Mean overall score across results, or None when there are none."""
    if not results:
        return None
    return round(sum(r.overall_score for r in results) / len(results), 3)


def _pass_rate(results: list[EvalResult]) -> float | None:
    """Fraction of results that passed the threshold, or None when there are none."""
    if not results:
        return None
    return round(sum(1 for r in results if r.passed) / len(results), 3)


def summarize_experiment(
    experiment: Experiment, results: list[EvalResult]
) -> ExperimentSummary:
    """Aggregate one experiment's stored results into headline metrics.

    Means and pass-rates are recomputed from ``results`` (the stored eval rows)
    so the summary is self-consistent even if results were re-scored; the delta
    is ``mean_v2 - mean_v1`` when both versions are present, else None.
    """
    v1 = _by_version(results, ResponseVersion.V1)
    v2 = _by_version(results, ResponseVersion.V2)
    mean_v1 = _mean(v1)
    mean_v2 = _mean(v2)
    delta = (
        round(mean_v2 - mean_v1, 3)
        if mean_v1 is not None and mean_v2 is not None
        else None
    )
    return ExperimentSummary(
        experiment_id=experiment.id,
        name=experiment.name,
        n_tasks=experiment.n_tasks,
        n_judged=experiment.n_judged,
        mean_v1=mean_v1,
        mean_v2=mean_v2,
        improvement_delta=delta,
        v1_pass_rate=_pass_rate(v1),
        v2_pass_rate=_pass_rate(v2),
    )


def per_criterion_means(results: list[EvalResult]) -> pd.DataFrame:
    """Mean score per criterion, split into v1/v2 columns.

    Combines deterministic and judge criteria; runs scored without the judge
    simply contribute no judge rows, so the frame holds whatever criteria are
    present (no assumption that the judge ran). Index is the criterion name;
    columns are the versions present ("v1", "v2"). Empty input → empty frame.
    """
    rows = [
        {"version": r.version.value, "criterion": s.name, "score": s.score}
        for r in results
        for s in (*r.deterministic, *r.judge)
    ]
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    table = frame.pivot_table(
        index="criterion", columns="version", values="score", aggfunc="mean"
    )
    return table.round(3)


def compare_versions(results: list[EvalResult]) -> VersionComparison:
    """Compare v1 vs v2: overall delta and per-criterion deltas.

    Per-criterion deltas are reported only for criteria present in *both*
    versions. When there are no v2 results the deltas are empty and
    ``overall_delta`` is None (a v1-only Day 3 run has nothing to compare).
    """
    v1 = _by_version(results, ResponseVersion.V1)
    v2 = _by_version(results, ResponseVersion.V2)
    mean_v1 = _mean(v1)
    mean_v2 = _mean(v2)
    overall_delta = (
        round(mean_v2 - mean_v1, 3)
        if mean_v1 is not None and mean_v2 is not None
        else None
    )

    per_criterion: dict[str, float] = {}
    table = per_criterion_means(results)
    if "v1" in table.columns and "v2" in table.columns:
        diff = (table["v2"] - table["v1"]).dropna()
        per_criterion = {name: round(float(d), 3) for name, d in diff.items()}

    return VersionComparison(
        mean_v1=mean_v1,
        mean_v2=mean_v2,
        overall_delta=overall_delta,
        per_criterion_delta=per_criterion,
    )


def experiment_trend(experiments: list[Experiment]) -> pd.DataFrame:
    """Build a chronological trend table across experiments for plotting.

    Columns: ``created_at``, ``name``, ``n_tasks``, ``mean_v1``, ``mean_v2``,
    ``improvement_delta``. Sorted oldest→newest so a trend line reads left to
    right. Empty input → empty frame.
    """
    if not experiments:
        return pd.DataFrame()
    rows = [
        {
            "created_at": e.created_at,
            "name": e.name,
            "n_tasks": e.n_tasks,
            "mean_v1": e.mean_v1,
            "mean_v2": e.mean_v2,
            "improvement_delta": e.improvement_delta,
        }
        for e in experiments
    ]
    return pd.DataFrame(rows).sort_values("created_at").reset_index(drop=True)


def check_regression(
    current_mean: float, baseline_mean: float | None, tolerance: float = 0.0
) -> RegressionResult:
    """Flag a regression when ``current_mean`` drops below the baseline.

    A regression is reported when ``current_mean < baseline_mean - tolerance``;
    a drop within ``tolerance`` is treated as noise and passes. With no baseline
    (the first experiment) there is nothing to regress against, so it passes.
    """
    if baseline_mean is None:
        return RegressionResult(
            regressed=False,
            current_mean=current_mean,
            message="No baseline experiment to compare against; passing.",
        )
    delta = round(current_mean - baseline_mean, 3)
    regressed = current_mean < baseline_mean - tolerance
    verdict = "REGRESSED" if regressed else "OK"
    return RegressionResult(
        regressed=regressed,
        current_mean=current_mean,
        baseline_mean=baseline_mean,
        delta=delta,
        tolerance=tolerance,
        message=f"{verdict}: {current_mean:.3f} vs baseline {baseline_mean:.3f} "
        f"(delta {delta:+.3f}, tolerance {tolerance:.3f}).",
    )
