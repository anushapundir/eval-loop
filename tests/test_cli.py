"""CLI wiring tests for main.py — argument parsing only (no model calls).

These guard that the Day 3 subcommands and their flags are wired correctly;
the evaluation logic itself is covered by the runner/judge tests.
"""

from __future__ import annotations

from main import build_parser


def test_parser_wires_evals_subcommand() -> None:
    args = build_parser().parse_args(
        ["evals", "--dataset", "synthetic", "--judge-sample-rate", "0.5"]
    )
    assert args.command == "evals"
    assert args.dataset == "synthetic"
    assert args.judge_sample_rate == 0.5
    assert args.no_judge is False


def test_parser_evals_defaults_and_no_judge_flag() -> None:
    args = build_parser().parse_args(["evals", "--no-judge"])
    assert args.dataset == "golden"  # default dataset
    assert args.judge_sample_rate is None  # falls back to settings default
    assert args.no_judge is True
    assert args.loop is False  # loop is opt-in; Day 3 behavior preserved by default
    assert args.split is None


def test_parser_wires_evals_loop_and_split() -> None:
    args = build_parser().parse_args(
        ["evals", "--dataset", "golden", "--loop", "--split", "test"]
    )
    assert args.command == "evals"
    assert args.loop is True
    assert args.split == "test"


def test_parser_wires_validate_judge_subcommand() -> None:
    args = build_parser().parse_args(["validate-judge", "--limit", "5"])
    assert args.command == "validate-judge"
    assert args.limit == 5


def test_parser_report_defaults() -> None:
    args = build_parser().parse_args(["report"])
    assert args.command == "report"
    assert args.experiment_id is None  # defaults to the most recent experiment
    assert args.gate is False  # gating is opt-in


def test_parser_wires_report_experiment_and_gate() -> None:
    args = build_parser().parse_args(["report", "--experiment-id", "abc123", "--gate"])
    assert args.command == "report"
    assert args.experiment_id == "abc123"
    assert args.gate is True
