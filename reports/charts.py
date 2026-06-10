"""Charts — the *pixels* half of the reporting layer (CLAUDE.md §8).

Renders the four Day 5 charts as PNGs into ``reports/output/``:

* score trend across experiments (mean v1/v2 over time),
* v1-vs-v2 overall comparison (the headline improvement),
* per-criterion breakdown (where the score comes from), and
* failure-mode counts (how failing responses fall short).

Each ``chart_*`` function is a pure renderer: it takes already-aggregated data
(from ``reports/metrics.py`` / ``evaluators/analysis.py``), writes one PNG, and
returns its path. ``render_all`` is a convenience that computes the pieces and
emits all four. The non-interactive Agg backend is selected before importing
pyplot so rendering works headless (Windows/CI, no display).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: must precede the pyplot import.

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from config.settings import Settings, get_settings  # noqa: E402
from evaluators.analysis import FailureAnalysis, cluster_failures  # noqa: E402
from reports.metrics import (  # noqa: E402
    ExperimentSummary,
    experiment_trend,
    per_criterion_means,
    summarize_experiment,
)
from storage.models import EvalResult, Experiment, ResponseVersion  # noqa: E402


def _resolve_dir(out_dir: Path | None, settings: Settings | None) -> Path:
    """Resolve and create the output directory (defaults to reports/output)."""
    out = out_dir or (settings or get_settings()).reports_output_dir
    out.mkdir(parents=True, exist_ok=True)
    return out


def _empty_figure(path: Path, message: str) -> Path:
    """Render a placeholder chart with a centered message, so a file always exists."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, color="gray")
    ax.axis("off")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_score_trend(trend: pd.DataFrame, out_dir: Path) -> Path:
    """Line chart of mean v1/v2 overall score across experiments (oldest→newest)."""
    path = out_dir / "score_trend.png"
    if trend.empty:
        return _empty_figure(path, "No experiments to plot.")

    x = range(len(trend))
    labels = [str(n) for n in trend["name"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    v1 = pd.to_numeric(trend["mean_v1"], errors="coerce")
    v2 = pd.to_numeric(trend["mean_v2"], errors="coerce")
    ax.plot(x, v1, marker="o", label="mean v1")
    if v2.notna().any():
        ax.plot(x, v2, marker="s", label="mean v2")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("overall score")
    ax.set_title("Score trend across experiments")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_v1_vs_v2(summary: ExperimentSummary, out_dir: Path) -> Path:
    """Bar chart of overall mean v1 vs v2 for one experiment, annotated with delta."""
    path = out_dir / "v1_vs_v2.png"
    if summary.mean_v1 is None:
        return _empty_figure(path, "No v1 results to plot.")

    labels = ["v1"]
    values = [summary.mean_v1]
    if summary.mean_v2 is not None:
        labels.append("v2")
        values.append(summary.mean_v2)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    bars = ax.bar(labels, values, color=["#888888", "#2a9d8f"][: len(labels)])
    for rect, val in zip(bars, values, strict=True):
        ax.text(rect.get_x() + rect.get_width() / 2, val + 0.02, f"{val:.3f}",
                ha="center", va="bottom")
    title = f"v1 vs v2 — {summary.name}"
    if summary.improvement_delta is not None:
        title += f"  (delta {summary.improvement_delta:+.3f})"
    ax.set_title(title)
    ax.set_ylabel("overall score")
    ax.set_ylim(0, 1)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_per_criterion(per_criterion: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar chart of mean score per criterion, by version."""
    path = out_dir / "per_criterion.png"
    if per_criterion.empty:
        return _empty_figure(path, "No criterion scores to plot.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    per_criterion.plot.bar(ax=ax)
    ax.set_ylabel("mean score")
    ax.set_xlabel("")
    ax.set_title("Per-criterion mean score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="version")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def chart_failure_modes(analysis: FailureAnalysis, out_dir: Path) -> Path:
    """Horizontal bar chart of failure-mode counts among failing results."""
    path = out_dir / "failure_modes.png"
    if not analysis.mode_counts:
        return _empty_figure(path, f"No failures ({analysis.n_total} results, all passed).")

    modes = list(analysis.mode_counts.keys())
    counts = list(analysis.mode_counts.values())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(modes, counts, color="#e76f51")
    ax.invert_yaxis()  # most common on top
    ax.set_xlabel("number of failing responses")
    ax.set_title(f"Failure modes ({analysis.n_failed}/{analysis.n_total} failed)")
    for i, c in enumerate(counts):
        ax.text(c + 0.05, i, str(c), va="center")
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def render_all(
    experiment: Experiment,
    results: list[EvalResult],
    all_experiments: list[Experiment],
    *,
    out_dir: Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Path]:
    """Compute the aggregates and render all four charts for one experiment.

    ``results`` are the focal experiment's eval results; ``all_experiments``
    drives the cross-experiment trend. Failure modes are computed over the v1
    results (what the agent gets wrong before revision). Returns a mapping of
    chart key → written path.
    """
    settings = settings or get_settings()
    out = _resolve_dir(out_dir, settings)

    summary = summarize_experiment(experiment, results)
    trend = experiment_trend(all_experiments)
    per_criterion = per_criterion_means(results)
    v1_results = [r for r in results if r.version is ResponseVersion.V1]
    failures = cluster_failures(v1_results, settings=settings)

    return {
        "score_trend": chart_score_trend(trend, out),
        "v1_vs_v2": chart_v1_vs_v2(summary, out),
        "per_criterion": chart_per_criterion(per_criterion, out),
        "failure_modes": chart_failure_modes(failures, out),
    }
