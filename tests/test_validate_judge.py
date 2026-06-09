"""Tests for judge validation (evaluators/validate_judge.py).

A fake judge (scores by length: long reference high, degraded variants low) lets
us verify the ranking math offline. The real validation makes Haiku calls and is
run from the CLI.
"""

from __future__ import annotations

from evaluators.validate_judge import build_bad_variants, format_report, validate_judge
from storage.models import CriterionScore, Task


def _fake_judge(task_prompt, text, context, *, settings=None):
    # Emulate a *quality* judge: only the full reference carries the marker word;
    # the truncated prefix and the off-topic banana text do not.
    s = 0.9 if "comfortably" in text else 0.2
    return [
        CriterionScore(name="correctness", score=s),
        CriterionScore(name="completeness", score=s),
        CriterionScore(name="clarity", score=s),
    ]


def _golden() -> list[Task]:
    return [
        Task(
            prompt=f"question {i}",
            source="golden",
            expected="A thorough, well grounded reference answer that comfortably clears the bar.",
        )
        for i in range(4)
    ]


def test_build_bad_variants_are_degraded() -> None:
    task = _golden()[0]
    variants = build_bad_variants(task)
    kinds = {v.kind for v in variants}
    assert "off_topic" in kinds
    assert "truncated" in kinds
    # Truncated text is much shorter than the reference.
    truncated = next(v for v in variants if v.kind == "truncated")
    assert len(truncated.text) < len(task.expected)


def test_validate_judge_ranks_good_over_bad(monkeypatch) -> None:
    report = validate_judge(
        _golden(), judge_fn=_fake_judge, context_fn=lambda q: "context", limit=4
    )
    assert report.n == 4
    assert report.ranking_accuracy == 1.0  # good beats bad on every task
    assert report.good_pass_rate == 1.0
    assert report.mean_good_score > report.mean_bad_score


def test_validate_judge_skips_tasks_without_reference() -> None:
    tasks = [Task(prompt="no label", source="golden")]  # expected is None
    report = validate_judge(tasks, judge_fn=_fake_judge, context_fn=lambda q: "c")
    assert report.n == 0


def test_format_report_mentions_n_and_accuracy() -> None:
    report = validate_judge(_golden(), judge_fn=_fake_judge, context_fn=lambda q: "c", limit=2)
    text = format_report(report)
    assert "N=2" in text
    assert "ranking accuracy" in text.lower()
