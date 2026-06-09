"""Inspect a stored improvement-loop run: show v1 -> feedback -> v2 for each task.

A read-only verification helper (it touches SQLite only through ``storage/db.py``,
the integration boundary). It prints a loop experiment's headline metrics and,
for every task the loop actually revised, the before/after answers, the feedback
that drove the revision, and the score delta.

Usage (from the repo root):

    python scripts/inspect_loop.py                 # latest loop experiment
    python scripts/inspect_loop.py --experiment <id>
    python scripts/inspect_loop.py --all           # include carried-forward tasks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make the repo root importable when run directly (python scripts/inspect_loop.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from storage import db  # noqa: E402  (import after sys.path bootstrap)
from storage.models import EvalResult, Experiment  # noqa: E402


def _latest_loop_experiment() -> Experiment | None:
    """Return the most recent experiment produced by a ``--loop`` run, or None."""
    for exp in db.list_experiments():  # newest first
        if exp.name.startswith("loop-") or exp.mean_v2 is not None:
            return exp
    return None


def _feedback_lines(task_id: str) -> list[str]:
    """Render the feedback-trace items for a task as ``[criterion] suggestion`` lines."""
    fb = next((t for t in db.list_traces(task_id=task_id) if t.step == "feedback"), None)
    if fb is None:
        return []
    return [f"  [{i['criterion']}] {i['suggestion']}" for i in fb.payload.get("items", [])]


def _print_pair(task_id: str, v1: EvalResult, v2: EvalResult) -> None:
    """Print one task's v1, feedback, v2, and score delta."""
    v1_resp = db.get_response(v1.response_id)
    v2_resp = db.get_response(v2.response_id)
    delta = round(v2.overall_score - v1.overall_score, 3)
    print("=" * 72)
    print(f"TASK {task_id}")
    print(f"\n--- v1 (score {v1.overall_score}) ---\n{v1_resp.text if v1_resp else '(missing)'}")
    lines = _feedback_lines(task_id)
    if lines:
        print("\n--- feedback ---")
        print("\n".join(lines))
    print(f"\n--- v2 (score {v2.overall_score}) ---\n{v2_resp.text if v2_resp else '(missing)'}")
    print(f"\n>>> delta = {delta:+}")


def main(argv: list[str] | None = None) -> int:
    """Print a loop experiment's revised (or all) v1->v2 pairs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", help="Experiment id (defaults to the latest loop run).")
    parser.add_argument(
        "--all", action="store_true",
        help="Show carried-forward tasks too, not just the ones that were revised.",
    )
    args = parser.parse_args(argv)

    if args.experiment:
        exp = next((e for e in db.list_experiments() if e.id == args.experiment), None)
    else:
        exp = _latest_loop_experiment()

    if exp is None:
        print("No loop experiment found. Run: "
              "python main.py evals --dataset golden --split test --loop --no-judge")
        return 1

    print(f"Experiment {exp.name}  ({exp.id})")
    print(f"  N={exp.n_tasks}  mean_v1={exp.mean_v1}  mean_v2={exp.mean_v2}  "
          f"delta={exp.improvement_delta}  judged={exp.n_judged}")
    print(f"  notes: {exp.notes}\n")

    by_task: dict[str, dict[str, EvalResult]] = {}
    for r in db.list_eval_results(experiment_id=exp.id):
        by_task.setdefault(r.task_id, {})[r.version.value] = r

    revised = 0
    for task_id, versions in by_task.items():
        v1, v2 = versions.get("v1"), versions.get("v2")
        if v1 is None or v2 is None:
            continue
        v1_resp, v2_resp = db.get_response(v1.response_id), db.get_response(v2.response_id)
        changed = bool(v1_resp and v2_resp and v1_resp.text != v2_resp.text)
        if changed:
            revised += 1
        if changed or args.all:
            _print_pair(task_id, v1, v2)

    print("=" * 72)
    print(f"Revised {revised}/{len(by_task)} task(s)."
          + ("" if args.all else "  (use --all to also see carried-forward tasks.)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
