"""CLI entry point for eval-loop-agent.

Day 1 commands:

* ``python main.py --hello`` — one local Ollama completion, logged (proves the
  model abstraction works).
* ``python main.py run "<task>"`` — run the agent on a task: generate a
  response, persist task/response/traces to SQLite, run the deterministic
  checks, store an EvalResult, and print a summary (the thin vertical slice).

Later days add ``evals`` and ``experiment`` subcommands.
"""

from __future__ import annotations

import argparse
import sys

from agents.graph import build_graph
from agents.llm import generate
from agents.state import AgentState
from config.logging import get_logger
from config.settings import get_settings
from evaluators.checks import run_basic_checks
from storage import db
from storage.models import EvalResult, Task

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
    """Run the agent on one task and persist the full slice to SQLite."""
    settings = get_settings()
    settings.ensure_dirs()
    db.init_db()

    task = Task(prompt=prompt, source="user")
    db.write_task(task)

    graph = build_graph()
    result = graph.invoke(AgentState(task=task, max_iterations=settings.max_iterations))
    state = AgentState(**result)

    if state.v1 is None:
        log.error("Agent produced no response.")
        return 1

    db.write_response(state.v1)
    for trace in state.traces:
        db.write_trace(trace)

    scores = run_basic_checks(state.v1.text)
    overall = round(sum(s.score for s in scores) / len(scores), 3)
    eval_result = EvalResult(
        task_id=task.id,
        response_id=state.v1.id,
        version=state.v1.version,
        deterministic=scores,
        overall_score=overall,
        passed=overall >= settings.pass_threshold,
    )
    db.write_eval_result(eval_result)

    retrieved = state.v1.retrieved_doc_ids
    print("\n=== Retrieved context ===")
    print(f"  docs: {', '.join(retrieved) if retrieved else '(none matched)'}")
    print("\n=== Response (v1) ===")
    print(state.v1.text)
    print("\n=== Deterministic checks ===")
    for s in scores:
        detail = f"  ({s.justification})" if s.justification else ""
        print(f"  {s.name:<10} {s.score:.2f}{detail}")
    print(f"  {'overall':<10} {overall:.2f}  ->  {'PASS' if eval_result.passed else 'FAIL'}"
          f" (threshold {settings.pass_threshold})")
    print(f"\nStored: task={task.id} response={state.v1.id} traces={len(state.traces)}")
    print(f"DB: {settings.db_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(prog="eval-loop-agent", description=__doc__)
    parser.add_argument("--hello", action="store_true", help="One local Ollama completion.")
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run", help="Run the agent on a task.")
    run_p.add_argument("task", help="The task prompt.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Dispatch CLI commands."""
    args = build_parser().parse_args(argv)
    if args.hello:
        return cmd_hello()
    if args.command == "run":
        return cmd_run(args.task)
    build_parser().print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
