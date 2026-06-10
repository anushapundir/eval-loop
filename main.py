"""CLI entry point for eval-loop-agent.

Day 1 commands:

* ``python main.py --hello`` — one local Ollama completion, logged (proves the
  model abstraction works).
* ``python main.py run "<task>"`` — run the agent on a task: generate a
  response, persist task/response/traces to SQLite, run the deterministic
  checks, store an EvalResult, and print a summary (the thin vertical slice).
* ``python main.py evals [--dataset golden|synthetic] [--judge-sample-rate R]
  [--no-judge]`` — score a frozen dataset offline: deterministic checks on all,
  the sampled Haiku judge on a subset, persisted as EvalResults + an Experiment.
* ``python main.py validate-judge [--limit N]`` — validate the judge against the
  golden set via good-vs-bad ranking and print an honest agreement report.
* ``python main.py report [--experiment-id ID] [--gate]`` — aggregate a stored
  experiment's results into metrics, write a JSON/CSV summary and four charts to
  ``reports/output/``, and optionally gate on regression vs the prior run.
"""

from __future__ import annotations

import argparse
import json
import sys

from agents.graph import build_graph
from agents.llm import generate
from agents.prompts import GENERATION_PROMPT_VERSION
from agents.state import AgentState
from config.logging import get_logger
from config.settings import get_settings
from datasets.loader import load_dataset
from evaluators.analysis import cluster_failures
from evaluators.runner import run_dataset, run_improvement
from evaluators.validate_judge import format_report, validate_judge
from reports.charts import render_all
from reports.metrics import (
    ExperimentSummary,
    check_regression,
    compare_versions,
    per_criterion_means,
    summarize_experiment,
)
from storage import db
from storage.models import EvalResult, Experiment, ResponseVersion, Task

log = get_logger("main")


def cmd_hello() -> int:
    """Make one local completion and log it; proves Ollama wiring."""
    settings = get_settings()
    log.info("Saying hello via provider=%s", settings.model_provider)
    completion = generate(
        "Reply with a single short sentence confirming you are working.",
        system="You are a terse assistant.",
    )
    log.info(
        "OK - %s/%s in %.0fms: %s",
        completion.provider, completion.model, completion.latency_ms, completion.text,
    )
    print(completion.text)
    return 0


def cmd_run(prompt: str) -> int:
    """Run the full critique->revise loop on one task and persist it to SQLite."""
    settings = get_settings()
    settings.ensure_dirs()
    db.init_db()

    task = Task(prompt=prompt, source="user")
    db.write_task(task)

    graph = build_graph()
    result = graph.invoke(AgentState(task=task, max_iterations=settings.max_iterations))
    state = AgentState(**result)

    if state.v1 is None or state.v2 is None:
        log.error("Agent produced an incomplete loop (v1=%s v2=%s).", state.v1, state.v2)
        return 1

    db.write_response(state.v1)
    db.write_response(state.v2)
    for trace in state.traces:
        db.write_trace(trace)
    if state.v1_eval is not None:
        db.write_eval_result(state.v1_eval)
    if state.v2_eval is not None:
        db.write_eval_result(state.v2_eval)

    retrieved = state.v1.retrieved_doc_ids
    revised = state.v2.text != state.v1.text
    print("\n=== Retrieved context ===")
    print(f"  docs: {', '.join(retrieved) if retrieved else '(none matched)'}")
    print("\n=== Response (v1) ===")
    print(state.v1.text)
    _print_scores("v1 checks", state.v1_eval, settings.pass_threshold)
    if revised:
        print("\n=== Feedback ===")
        print(state.feedback or "(none)")
        print("\n=== Response (v2, revised) ===")
        print(state.v2.text)
        _print_scores("v2 checks", state.v2_eval, settings.pass_threshold)
        d_v1 = state.v1_eval.overall_score if state.v1_eval else 0.0
        d_v2 = state.v2_eval.overall_score if state.v2_eval else 0.0
        print(f"\n  delta (v2-v1): {d_v2 - d_v1:+.3f}")
    else:
        print("\n(v1 already passed — carried forward unchanged as v2.)")
    print(f"\nStored: task={task.id} v1={state.v1.id} v2={state.v2.id} "
          f"traces={len(state.traces)}")
    print(f"DB: {settings.db_path}")
    return 0


def _print_scores(title: str, eval_result: EvalResult | None, threshold: float) -> None:
    """Print a deterministic EvalResult's per-criterion scores and overall verdict."""
    if eval_result is None:
        return
    print(f"\n=== {title} ===")
    for s in eval_result.deterministic:
        detail = f"  ({s.justification})" if s.justification else ""
        print(f"  {s.name:<10} {s.score:.2f}{detail}")
    verdict = "PASS" if eval_result.passed else "FAIL"
    print(f"  {'overall':<10} {eval_result.overall_score:.2f}  ->  {verdict}"
          f" (threshold {threshold})")


def cmd_evals(
    dataset: str,
    judge_sample_rate: float | None,
    no_judge: bool,
    loop: bool,
    split: str | None,
) -> int:
    """Score a frozen dataset offline and persist results + an experiment record.

    Without ``--loop`` this is the Day 3 v1-only evaluation. With ``--loop`` it
    runs the full critique->revise loop per task and reports mean v1 vs mean v2
    plus the improvement delta (Day 4). ``--split`` selects the golden dev/test
    partition (ignored for the synthetic set, which has no split).
    """
    settings = get_settings()
    settings.ensure_dirs()
    db.init_db()

    path = (
        settings.golden_dir / "golden.jsonl"
        if dataset == "golden"
        else settings.synthetic_dir / "synthetic.jsonl"
    )
    # Split only applies to the golden set; the synthetic set carries no labels.
    effective_split = split if dataset == "golden" else None
    tasks = load_dataset(path, split=effective_split)
    if not tasks:
        log.error("No tasks loaded from %s (split=%s)", path, effective_split)
        return 1

    # Effective judge sample rate: --no-judge wins, else --judge-sample-rate, else default.
    if no_judge:
        rate = 0.0
    elif judge_sample_rate is not None:
        rate = judge_sample_rate
    else:
        rate = settings.judge_sample_rate
    run_settings = settings.model_copy(update={"judge_sample_rate": rate})

    experiment = Experiment(
        name=f"{'loop' if loop else 'eval'}-{dataset}",
        prompt_version=GENERATION_PROMPT_VERSION,
        model_provider=settings.model_provider,
        n_tasks=len(tasks),
    )

    graph = build_graph()

    if loop:
        return _run_loop_eval(
            dataset, effective_split, tasks, graph, experiment, run_settings, settings, rate
        )

    def responder(task: Task) -> tuple:
        result = graph.invoke(AgentState(task=task, max_iterations=settings.max_iterations))
        state = AgentState(**result)
        return state.v1, state.context

    results = run_dataset(
        tasks, responder, experiment_id=experiment.id, settings=run_settings
    )

    n_judged = sum(1 for r in results if r.judged)
    n_passed = sum(1 for r in results if r.passed)
    mean_overall = round(sum(r.overall_score for r in results) / len(results), 3)
    experiment.n_judged = n_judged
    experiment.mean_v1 = mean_overall  # Day 3 path scores v1 only.
    experiment.notes = f"Day 3 offline eval on {dataset} (N={len(results)})."
    db.write_experiment(experiment)

    print(f"\n=== Evaluation: {dataset} (N={len(results)}) ===")
    print(f"  mean overall : {mean_overall:.3f}")
    print(f"  pass rate    : {n_passed}/{len(results)} ({n_passed / len(results):.2f}) "
          f"at threshold {settings.pass_threshold}")
    print(f"  judged       : {n_judged}/{len(results)} "
          f"(rate={rate:.2f}, seed={settings.judge_sample_seed})")
    print(f"  experiment   : {experiment.id}")
    print(f"DB: {settings.db_path}")
    return 0


def _run_loop_eval(dataset, split, tasks, graph, experiment, run_settings, settings, rate) -> int:
    """Run the Day 4 critique->revise loop over the dataset and report v1 vs v2."""

    def loop_runner(task: Task) -> AgentState:
        result = graph.invoke(AgentState(task=task, max_iterations=settings.max_iterations))
        return AgentState(**result)

    imp = run_improvement(
        tasks, loop_runner, experiment_id=experiment.id, settings=run_settings
    )

    n = len(imp.v1_results)
    experiment.n_judged = imp.n_judged
    experiment.mean_v1 = imp.mean_v1
    experiment.mean_v2 = imp.mean_v2
    experiment.improvement_delta = imp.improvement_delta
    experiment.notes = f"Day 4 loop on {dataset} split={split or 'all'} (N={n})."
    db.write_experiment(experiment)

    improved = sum(1 for a, b in zip(imp.v1_results, imp.v2_results, strict=True)
                   if b.overall_score > a.overall_score)
    regressed = sum(1 for a, b in zip(imp.v1_results, imp.v2_results, strict=True)
                    if b.overall_score < a.overall_score)
    arrow = "+" if imp.improvement_delta >= 0 else ""
    print(f"\n=== Improvement loop: {dataset} split={split or 'all'} (N={n}) ===")
    print(f"  mean v1      : {imp.mean_v1:.3f}")
    print(f"  mean v2      : {imp.mean_v2:.3f}")
    print(f"  delta (v2-v1): {arrow}{imp.improvement_delta:.3f}")
    print(f"  improved     : {improved}/{n}  regressed: {regressed}/{n}  "
          f"unchanged: {n - improved - regressed}/{n}")
    print(f"  judged       : {imp.n_judged}/{n} "
          f"(rate={rate:.2f}, seed={settings.judge_sample_seed})")
    print(f"  experiment   : {experiment.id}")
    print(f"DB: {settings.db_path}")
    return 0


def cmd_validate_judge(limit: int) -> int:
    """Validate the judge against the golden set and print the agreement report."""
    settings = get_settings()
    tasks = load_dataset(settings.golden_dir / "golden.jsonl")
    report = validate_judge(tasks, settings=settings, limit=limit)
    print(format_report(report))
    return 0


def _fmt(x: float | None) -> str:
    """Format an optional score for display."""
    return f"{x:.3f}" if x is not None else "n/a"


def cmd_report(experiment_id: str | None, gate: bool) -> int:
    """Aggregate a stored experiment into metrics + charts, with an optional gate.

    Reads only from SQLite (no model calls), so it is free and reproducible. With
    no ``--experiment-id`` it reports the newest experiment. ``--gate`` compares
    this run's headline mean against the prior experiment and exits non-zero on a
    regression beyond ``settings.regression_tolerance``.
    """
    settings = get_settings()
    settings.ensure_dirs()
    db.init_db()

    experiments = db.list_experiments()  # newest first
    if not experiments:
        log.error("No experiments in %s; run `evals` first.", settings.db_path)
        return 1

    if experiment_id:
        focal = next((e for e in experiments if e.id == experiment_id), None)
        if focal is None:
            log.error("Experiment %s not found.", experiment_id)
            return 1
    else:
        focal = experiments[0]

    results = db.list_eval_results(focal.id)
    if not results:
        log.error("Experiment %s has no eval results to report.", focal.id)
        return 1

    summary = summarize_experiment(focal, results)
    comparison = compare_versions(results)
    v1_results = [r for r in results if r.version is ResponseVersion.V1]
    failures = cluster_failures(v1_results, settings=settings)

    # Write artifacts: a JSON summary, a per-criterion CSV, and the four charts.
    out = settings.reports_output_dir
    summary_path = out / "report_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "summary": summary.model_dump(),
                "comparison": comparison.model_dump(),
                "failure_modes": failures.model_dump(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_path = out / "per_criterion.csv"
    per_criterion_means(results).to_csv(csv_path)
    charts = render_all(focal, results, experiments, settings=settings)

    print(f"\n=== Report: {focal.name} (N={focal.n_tasks}) ===")
    print(f"  experiment   : {focal.id}")
    print(f"  mean v1      : {_fmt(summary.mean_v1)}")
    print(f"  mean v2      : {_fmt(summary.mean_v2)}")
    if summary.improvement_delta is not None:
        print(f"  delta (v2-v1): {summary.improvement_delta:+.3f}")
    print(f"  v1 pass rate : {_fmt(summary.v1_pass_rate)}")
    if summary.v2_pass_rate is not None:
        print(f"  v2 pass rate : {_fmt(summary.v2_pass_rate)}")
    print(f"  judged       : {summary.n_judged}/{summary.n_tasks}")
    if failures.mode_counts:
        modes = ", ".join(f"{k}={v}" for k, v in failures.mode_counts.items())
        print(f"  failure modes: {modes} ({failures.n_failed}/{failures.n_total} v1 failed)")
    print("\n  charts:")
    for key, path in charts.items():
        print(f"    {key:<14} {path}")
    print(f"  summary json : {summary_path}")
    print(f"  per-criterion: {csv_path}")

    exit_code = _run_gate(focal, summary, experiments, settings) if gate else 0
    print(f"\nDB: {settings.db_path}")
    return exit_code


def _run_gate(focal, summary: ExperimentSummary, experiments, settings) -> int:
    """Compare the focal run's headline mean against the prior experiment."""
    current = summary.mean_v2 if summary.mean_v2 is not None else summary.mean_v1
    prior = [e for e in experiments if e.created_at < focal.created_at]
    baseline_exp = max(prior, key=lambda e: e.created_at) if prior else None
    baseline = None
    if baseline_exp is not None:
        baseline = (
            baseline_exp.mean_v2 if baseline_exp.mean_v2 is not None else baseline_exp.mean_v1
        )
    result = check_regression(current, baseline, tolerance=settings.regression_tolerance)
    print("\n=== Regression gate ===")
    if baseline_exp is not None:
        print(f"  baseline     : {baseline_exp.name} ({baseline_exp.id})")
    print(f"  {result.message}")
    return 1 if result.regressed else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="eval-loop-agent", description=__doc__)
    parser.add_argument("--hello", action="store_true", help="One local Ollama completion.")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run", help="Run the agent on a task.")
    run_p.add_argument("task", help="The task prompt.")

    evals_p = sub.add_parser("evals", help="Evaluate a frozen dataset offline.")
    evals_p.add_argument("--dataset", choices=["golden", "synthetic"], default="golden")
    evals_p.add_argument(
        "--judge-sample-rate", type=float, default=None,
        help="Fraction of responses judged (defaults to settings.judge_sample_rate).",
    )
    evals_p.add_argument(
        "--no-judge", action="store_true", help="Deterministic checks only (no paid judge)."
    )
    evals_p.add_argument(
        "--loop", action="store_true",
        help="Run the full critique->revise loop and compare v1 vs v2 (Day 4).",
    )
    evals_p.add_argument(
        "--split", choices=["dev", "test"], default=None,
        help="Golden split to evaluate (dev to develop, test for the held-out headline).",
    )

    vj_p = sub.add_parser("validate-judge", help="Validate the judge on the golden set.")
    vj_p.add_argument(
        "--limit", type=int, default=8, help="Max golden tasks to probe (controls cost)."
    )

    report_p = sub.add_parser("report", help="Aggregate an experiment into metrics + charts.")
    report_p.add_argument(
        "--experiment-id", default=None,
        help="Experiment to report (defaults to the most recent).",
    )
    report_p.add_argument(
        "--gate", action="store_true",
        help="Exit non-zero if the run regresses vs the prior experiment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch CLI commands."""
    args = build_parser().parse_args(argv)
    if args.hello:
        return cmd_hello()
    if args.command == "run":
        return cmd_run(args.task)
    if args.command == "evals":
        return cmd_evals(
            args.dataset, args.judge_sample_rate, args.no_judge, args.loop, args.split
        )
    if args.command == "validate-judge":
        return cmd_validate_judge(args.limit)
    if args.command == "report":
        return cmd_report(args.experiment_id, args.gate)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
