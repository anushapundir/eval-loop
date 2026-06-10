"""Failure-mode analysis — cluster failing results into named modes (CLAUDE.md §5/§7).

Knowing the mean score tells us *that* responses fall short; this tells us *how*.
For every failing :class:`EvalResult`, we look at which criteria dropped below the
threshold and attribute the failure to one or more named modes (e.g. a low
``grounding`` score → ``low_grounding``). The counts feed the failure-mode chart
and point the next improvement iteration at the right problem.

Deterministic and free: it reads the per-criterion scores already stored on each
result — no model calls.
"""

from __future__ import annotations

from pydantic import BaseModel

from config.settings import Settings, get_settings
from storage.models import EvalResult

# Map each criterion (from checks.py + rubric.py) to a human-named failure mode.
# Unknown criteria fall back to their own name (see ``_mode_for``).
_MODE_BY_CRITERION: dict[str, str] = {
    "non_empty": "empty_response",
    "length": "length_violation",
    "grounding": "low_grounding",
    "coverage": "missing_coverage",
    "correctness": "incorrect",
    "completeness": "incomplete",
    "clarity": "unclear",
}


class FailureAnalysis(BaseModel):
    """Counts of named failure modes across a set of results."""

    n_total: int
    n_failed: int
    # mode name -> number of failed results exhibiting that mode (a result can
    # exhibit several, so the counts need not sum to ``n_failed``).
    mode_counts: dict[str, int] = {}


def _mode_for(criterion_name: str) -> str:
    """Map a criterion name to its named failure mode (fallback: the name itself)."""
    return _MODE_BY_CRITERION.get(criterion_name, criterion_name)


def cluster_failures(
    results: list[EvalResult],
    *,
    threshold: float | None = None,
    settings: Settings | None = None,
) -> FailureAnalysis:
    """Cluster failing results by which criteria fell below the threshold.

    A result counts as failing when ``passed`` is False. Within each failure,
    every criterion (deterministic or judge) scoring below ``threshold`` is
    attributed to its named mode. A failure whose individual criteria all clear
    the threshold (overall dragged down by the blend) is recorded as ``other``.

    Args:
        results: Eval results to analyze (e.g. the v1 results of an experiment).
        threshold: Per-criterion weakness cutoff; defaults to ``pass_threshold``.
        settings: Injectable settings (defaults to the cached singleton).

    Returns:
        A :class:`FailureAnalysis` with totals and per-mode counts.
    """
    settings = settings or get_settings()
    cutoff = threshold if threshold is not None else settings.pass_threshold

    failed = [r for r in results if not r.passed]
    mode_counts: dict[str, int] = {}
    for result in failed:
        weak = {
            _mode_for(s.name)
            for s in (*result.deterministic, *result.judge)
            if s.score < cutoff
        }
        if not weak:
            weak = {"other"}
        for mode in weak:
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    return FailureAnalysis(
        n_total=len(results),
        n_failed=len(failed),
        mode_counts=dict(sorted(mode_counts.items(), key=lambda kv: (-kv[1], kv[0]))),
    )
