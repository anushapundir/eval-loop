"""Evaluation runner — orchestrates checks + sampled judge into EvalResults.

This is the seam where the evaluation framework comes together (CLAUDE.md §7):
deterministic checks run on *every* response (free), and the paid judge runs on
a reproducibly sampled subset (``judge_sample_rate`` + ``judge_sample_seed``).
Results are persisted through ``storage/db.py`` — the integration boundary — so
the dataset can be re-scored offline without re-running the agent.

The runner takes a ``responder`` callable instead of importing the graph, which
keeps it decoupled and testable offline (the CLI wires in a graph-based one).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from config.logging import get_logger
from config.settings import Settings, get_settings
from evaluators.checks import run_deterministic_checks
from evaluators.judge import judge_response
from storage import db
from storage.models import AgentResponse, EvalResult, Task

if TYPE_CHECKING:
    from agents.state import AgentState

log = get_logger(__name__)

# A responder turns a task into (response, retrieved_context). The runner stays
# agnostic to *how* the response was produced (live graph, stored row, fake).
Responder = Callable[[Task], tuple[AgentResponse, str]]

# A loop runner takes a task through the full critique-and-revise graph and
# returns the final state (carrying v1, v2, and their context).
LoopRunner = Callable[[Task], "AgentState"]


def select_judge_indices(n: int, rate: float, seed: int) -> set[int]:
    """Pick which of ``n`` responses get the paid judge, reproducibly.

    Count is ``round(n * rate)``; the specific indices are chosen with a seeded
    RNG so the same dataset and seed always sample the same subset.
    """
    count = round(n * rate)
    if count <= 0:
        return set()
    if count >= n:
        return set(range(n))
    return set(random.Random(seed).sample(range(n), count))


def evaluate_response(
    response: AgentResponse,
    task: Task,
    context: str,
    *,
    do_judge: bool,
    settings: Settings | None = None,
    experiment_id: str | None = None,
) -> EvalResult:
    """Score one response: deterministic checks always, judge when ``do_judge``.

    ``overall_score`` is the mean of the deterministic checks, and when the judge
    ran it is an equal blend of the deterministic mean and the judge mean
    (0.5/0.5). ``judged`` is True only if the judge actually returned scores, so
    a degraded judge call leaves a clean deterministic-only result.
    """
    settings = settings or get_settings()

    deterministic = run_deterministic_checks(
        response.text, context=context, key_points=task.key_points
    )
    judge = (
        judge_response(task.prompt, response.text, context, settings=settings)
        if do_judge
        else []
    )
    judged = bool(judge)

    overall = _overall_score(deterministic, judge)
    return EvalResult(
        task_id=task.id,
        response_id=response.id,
        version=response.version,
        deterministic=deterministic,
        judge=judge,
        overall_score=overall,
        passed=overall >= settings.pass_threshold,
        judged=judged,
        experiment_id=experiment_id,
    )


def run_dataset(
    tasks: list[Task],
    responder: Responder,
    *,
    experiment_id: str | None = None,
    settings: Settings | None = None,
    db_path: Path | None = None,
    persist: bool = True,
) -> list[EvalResult]:
    """Evaluate every task, sampling the judge, and persist the results.

    Args:
        tasks: The dataset to score.
        responder: Produces ``(response, context)`` for each task.
        experiment_id: Tag results to an experiment (optional).
        settings: Injectable settings (defaults to the cached singleton).
        db_path: Target SQLite DB (defaults to ``settings.db_path``).
        persist: Write task/response/eval rows (False for dry runs/tests).

    Returns:
        One :class:`EvalResult` per task, in order.
    """
    settings = settings or get_settings()
    judge_indices = select_judge_indices(
        len(tasks), settings.judge_sample_rate, settings.judge_sample_seed
    )
    log.info(
        "Evaluating %d tasks; judging %d (rate=%.2f, seed=%d), rest deterministic-only.",
        len(tasks), len(judge_indices), settings.judge_sample_rate, settings.judge_sample_seed,
    )

    results: list[EvalResult] = []
    for i, task in enumerate(tasks):
        response, context = responder(task)
        result = evaluate_response(
            response,
            task,
            context,
            do_judge=i in judge_indices,
            settings=settings,
            experiment_id=experiment_id,
        )
        if persist:
            db.write_task(task, db_path)
            db.write_response(response, db_path)
            db.write_eval_result(result, db_path)
        results.append(result)
    return results


def _overall_score(deterministic, judge) -> float:
    """Mean of deterministic scores, blended 50/50 with the judge mean if judged."""
    det_mean = sum(s.score for s in deterministic) / len(deterministic)
    if not judge:
        return round(det_mean, 3)
    judge_mean = sum(s.score for s in judge) / len(judge)
    return round(0.5 * det_mean + 0.5 * judge_mean, 3)


@dataclass(frozen=True)
class ImprovementResults:
    """The outcome of running the loop over a dataset: v1 vs v2, scored and compared."""

    v1_results: list[EvalResult]
    v2_results: list[EvalResult]
    mean_v1: float
    mean_v2: float
    improvement_delta: float
    n_judged: int


def run_improvement(
    tasks: list[Task],
    loop_runner: LoopRunner,
    *,
    experiment_id: str | None = None,
    settings: Settings | None = None,
    db_path: Path | None = None,
    persist: bool = True,
) -> ImprovementResults:
    """Run the critique-and-revise loop over a dataset and compare v1 vs v2.

    Each task goes through ``loop_runner`` (the full graph), yielding v1 and v2.
    Both are scored with :func:`evaluate_response`, and the paid judge — when
    sampled — is applied to the *same* task indices for both versions, so the
    judged comparison is fair. Deterministic checks always run (free).

    Returns aggregate means and the improvement delta (mean_v2 − mean_v1).
    """
    settings = settings or get_settings()
    judge_indices = select_judge_indices(
        len(tasks), settings.judge_sample_rate, settings.judge_sample_seed
    )
    log.info(
        "Improvement loop over %d tasks; judging %d each version (rate=%.2f, seed=%d).",
        len(tasks), len(judge_indices), settings.judge_sample_rate, settings.judge_sample_seed,
    )

    v1_results: list[EvalResult] = []
    v2_results: list[EvalResult] = []
    for i, task in enumerate(tasks):
        state = loop_runner(task)
        if state.v1 is None or state.v2 is None:
            log.warning("Loop produced an incomplete state for task %s; skipping.", task.id)
            continue
        do_judge = i in judge_indices
        v1 = evaluate_response(
            state.v1, task, state.context,
            do_judge=do_judge, settings=settings, experiment_id=experiment_id,
        )
        v2 = evaluate_response(
            state.v2, task, state.context,
            do_judge=do_judge, settings=settings, experiment_id=experiment_id,
        )
        if persist:
            db.write_task(task, db_path)
            db.write_response(state.v1, db_path)
            db.write_response(state.v2, db_path)
            for trace in state.traces:
                db.write_trace(trace, db_path)
            db.write_eval_result(v1, db_path)
            db.write_eval_result(v2, db_path)
        v1_results.append(v1)
        v2_results.append(v2)

    mean_v1 = _mean_overall(v1_results)
    mean_v2 = _mean_overall(v2_results)
    return ImprovementResults(
        v1_results=v1_results,
        v2_results=v2_results,
        mean_v1=mean_v1,
        mean_v2=mean_v2,
        improvement_delta=round(mean_v2 - mean_v1, 3),
        n_judged=sum(1 for r in v1_results if r.judged),
    )


def _mean_overall(results: list[EvalResult]) -> float:
    """Mean overall score across results, or 0.0 when empty."""
    if not results:
        return 0.0
    return round(sum(r.overall_score for r in results) / len(results), 3)
