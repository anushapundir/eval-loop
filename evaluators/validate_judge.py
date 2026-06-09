"""Judge validation — the non-negotiable trust check (CLAUDE.md §7).

Before any judge score is trusted, we validate the judge against the
human-verified golden set using a *good-vs-bad ranking* probe: for each golden
task we score the reference answer (known-good) and two deliberately degraded
variants (off-topic and truncated), then check that the judge ranks the good
answer above the bad ones and that good answers clear the pass threshold.

We report a single ranking-accuracy number and N honestly — at small N this is
directional, not production-grade significance. If accuracy is low, the fix is
to refine ``rubric.py`` and re-run; the rubric is the tunable knob.

This module is pure given an injected ``judge_fn``/``context_fn`` (so it tests
offline); the CLI wires in the real Haiku judge and KB retrieval.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from config.settings import Settings, get_settings
from evaluators.judge import judge_response
from storage.models import CriterionScore, Task

# Clearly irrelevant text — a correct judge should score this low on every task.
_OFF_TOPIC = "Bananas are a good source of potassium and grow in tropical climates."

JudgeFn = Callable[..., list[CriterionScore]]
ContextFn = Callable[[str], str]


@dataclass(frozen=True)
class Probe:
    """One answer to score during validation, with its expected label."""

    text: str
    label: str  # "good" | "bad"
    kind: str  # "reference" | "off_topic" | "truncated"


@dataclass(frozen=True)
class ValidationReport:
    """The outcome of validating the judge on the golden probe set."""

    n: int  # number of golden tasks probed (each has 1 good + 2 bad answers)
    ranking_accuracy: float  # fraction where good scored above the best bad
    good_pass_rate: float  # fraction where the good answer cleared the threshold
    mean_good_score: float
    mean_bad_score: float


def build_bad_variants(task: Task) -> list[Probe]:
    """Two degraded variants of a task's reference answer."""
    reference = task.expected or ""
    return [
        Probe(text=_OFF_TOPIC, label="bad", kind="off_topic"),
        Probe(text=reference[:15], label="bad", kind="truncated"),
    ]


def validate_judge(
    tasks: list[Task],
    *,
    settings: Settings | None = None,
    judge_fn: JudgeFn = judge_response,
    context_fn: ContextFn | None = None,
    limit: int | None = None,
) -> ValidationReport:
    """Validate the judge by good-vs-bad ranking over labeled golden tasks.

    Args:
        tasks: Golden tasks; only those with an ``expected`` reference are used.
        settings: Injectable settings (defaults to the cached singleton).
        judge_fn: The judge to validate (defaults to the real Haiku judge).
        context_fn: Maps a task prompt to retrieved context (defaults to KB
            retrieval); injected as a no-op-ish stub in tests.
        limit: Cap the number of tasks probed, to control cost.

    Returns:
        A :class:`ValidationReport` with ranking accuracy and pass rate.
    """
    settings = settings or get_settings()
    context_fn = context_fn or _default_context_fn(settings)

    labeled = [t for t in tasks if t.expected]
    if limit is not None:
        labeled = labeled[:limit]

    good_scores: list[float] = []
    bad_scores: list[float] = []
    ranking_correct = 0
    good_passes = 0

    for task in labeled:
        context = context_fn(task.prompt)
        good = _judge_mean(judge_fn(task.prompt, task.expected, context, settings=settings))
        bads = [
            _judge_mean(judge_fn(task.prompt, v.text, context, settings=settings))
            for v in build_bad_variants(task)
        ]
        best_bad = max(bads)

        good_scores.append(good)
        bad_scores.append(best_bad)
        if good > best_bad:
            ranking_correct += 1
        if good >= settings.pass_threshold:
            good_passes += 1

    n = len(labeled)
    return ValidationReport(
        n=n,
        ranking_accuracy=_safe_div(ranking_correct, n),
        good_pass_rate=_safe_div(good_passes, n),
        mean_good_score=round(_mean(good_scores), 3),
        mean_bad_score=round(_mean(bad_scores), 3),
    )


def format_report(report: ValidationReport) -> str:
    """Render a validation report as a short, honest summary."""
    return (
        f"Judge validation (good-vs-bad ranking), N={report.n}:\n"
        f"  ranking accuracy : {report.ranking_accuracy:.2f}  "
        f"(good scored above the best bad variant)\n"
        f"  good pass rate   : {report.good_pass_rate:.2f}  "
        f"(reference answers clearing threshold)\n"
        f"  mean good score  : {report.mean_good_score:.2f}\n"
        f"  mean bad score   : {report.mean_bad_score:.2f}\n"
        f"  Note: small-N validation is directional, not statistically significant."
    )


def _default_context_fn(settings: Settings) -> ContextFn:
    """Build a context function that retrieves KB context for a prompt."""
    from agents.tools import load_kb, retrieve

    chunks = load_kb(settings.kb_dir)

    def _ctx(prompt: str) -> str:
        return retrieve(prompt, chunks, k=settings.retrieval_top_k).context

    return _ctx


def _judge_mean(scores: list[CriterionScore]) -> float:
    """Mean of judge criterion scores; 0.0 if the judge degraded to nothing."""
    return sum(s.score for s in scores) / len(scores) if scores else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0
