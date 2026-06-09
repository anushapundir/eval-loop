"""Tests for the evaluation runner (evaluators/runner.py).

The runner is tested offline: a fake responder supplies (response, context) so no
Ollama is needed, and the judge is monkeypatched so no Haiku call is made.
Persistence is checked against a temp SQLite DB.
"""

from __future__ import annotations

from pathlib import Path

from agents.state import AgentState
from config.settings import Settings
from evaluators import runner as runner_module
from evaluators.runner import (
    evaluate_response,
    run_dataset,
    run_improvement,
    select_judge_indices,
)
from storage import db
from storage.models import AgentResponse, CriterionScore, ResponseVersion, Task


def _settings(**overrides) -> Settings:
    base = dict(judge_sample_rate=0.0, judge_sample_seed=42, pass_threshold=0.7)
    base.update(overrides)
    return Settings(**base)


def _responder(task: Task):
    resp = AgentResponse(
        task_id=task.id, text="grounded answer about retrieval", retrieved_doc_ids=["rag"]
    )
    return resp, "[rag] retrieval grounding context"


# --- sampling ---------------------------------------------------------------


def test_select_judge_indices_is_deterministic() -> None:
    first = select_judge_indices(50, rate=0.2, seed=42)
    second = select_judge_indices(50, rate=0.2, seed=42)
    assert first == second
    assert len(first) == 10  # round(50 * 0.2)


def test_select_judge_indices_edges() -> None:
    assert select_judge_indices(20, rate=0.0, seed=1) == set()
    assert select_judge_indices(20, rate=1.0, seed=1) == set(range(20))


# --- single-response evaluation --------------------------------------------


def test_evaluate_response_deterministic_only() -> None:
    task = Task(prompt="what is retrieval grounding", source="golden", key_points=["retrieval"])
    resp = AgentResponse(task_id=task.id, text="retrieval grounding context answer")
    result = evaluate_response(
        resp, task, "[rag] retrieval grounding context", do_judge=False, settings=_settings()
    )

    assert result.judged is False
    assert result.judge == []
    assert [s.name for s in result.deterministic] == [
        "non_empty", "length", "grounding", "coverage",
    ]
    det_mean = sum(s.score for s in result.deterministic) / 4
    assert result.overall_score == round(det_mean, 3)


def test_evaluate_response_blends_judge_when_judged(monkeypatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "judge_response",
        lambda *a, **k: [
            CriterionScore(name="correctness", score=1.0),
            CriterionScore(name="completeness", score=1.0),
            CriterionScore(name="clarity", score=1.0),
        ],
    )
    task = Task(prompt="what is retrieval grounding", key_points=["retrieval"])
    resp = AgentResponse(task_id=task.id, text="retrieval grounding context answer")
    result = evaluate_response(
        resp, task, "[rag] retrieval grounding context", do_judge=True, settings=_settings()
    )

    assert result.judged is True
    assert len(result.judge) == 3
    det_mean = sum(s.score for s in result.deterministic) / 4
    assert result.overall_score == round(0.5 * det_mean + 0.5 * 1.0, 3)


def test_evaluate_response_not_judged_when_judge_returns_empty(monkeypatch) -> None:
    """If the judge degrades to [], the result is deterministic-only and judged=False."""
    monkeypatch.setattr(runner_module, "judge_response", lambda *a, **k: [])
    task = Task(prompt="x", key_points=[])
    resp = AgentResponse(task_id=task.id, text="some sufficiently long answer text here")
    result = evaluate_response(resp, task, "ctx", do_judge=True, settings=_settings())

    assert result.judged is False
    assert result.judge == []


# --- dataset run + persistence ---------------------------------------------


def test_run_dataset_persists_and_samples(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runner_module,
        "judge_response",
        lambda *a, **k: [CriterionScore(name="correctness", score=0.5),
                         CriterionScore(name="completeness", score=0.5),
                         CriterionScore(name="clarity", score=0.5)],
    )
    dbp = tmp_path / "run.db"
    db.init_db(dbp)
    tasks = [Task(prompt=f"q{i}", source="synthetic") for i in range(10)]

    results = run_dataset(
        tasks, _responder, settings=_settings(judge_sample_rate=0.2), db_path=dbp
    )

    assert len(results) == 10
    # 2 of 10 judged (round(10 * 0.2)); the rest deterministic-only.
    assert sum(1 for r in results if r.judged) == 2
    # Every result and its task/response persisted.
    stored = db.list_eval_results(db_path=dbp)
    assert len(stored) == 10
    assert all(r.version is ResponseVersion.V1 for r in stored)


# --- improvement loop (v1 vs v2) -------------------------------------------


def _loop_runner(task: Task) -> AgentState:
    """A fake graph run: weak v1, stronger v2, both grounded in the same context."""
    v1 = AgentResponse(task_id=task.id, version=ResponseVersion.V1, text="weak short answer")
    v2 = AgentResponse(
        task_id=task.id,
        version=ResponseVersion.V2,
        text="retrieval grounding context is the improved grounded answer",
    )
    return AgentState(task=task, context="[rag] retrieval grounding context", v1=v1, v2=v2)


def test_run_improvement_computes_means_and_delta(tmp_path: Path) -> None:
    dbp = tmp_path / "imp.db"
    db.init_db(dbp)
    tasks = [Task(prompt="retrieval grounding context", source="golden") for _ in range(4)]

    imp = run_improvement(
        tasks, _loop_runner, settings=_settings(judge_sample_rate=0.0), db_path=dbp
    )

    assert len(imp.v1_results) == 4 and len(imp.v2_results) == 4
    expected_delta = round(imp.mean_v2 - imp.mean_v1, 3)
    assert imp.improvement_delta == expected_delta
    # The v2 answer is more grounded/longer than v1, so it should not score lower.
    assert imp.mean_v2 >= imp.mean_v1
    # Both versions persisted, tagged by version.
    stored = db.list_eval_results(db_path=dbp)
    assert sum(1 for r in stored if r.version is ResponseVersion.V1) == 4
    assert sum(1 for r in stored if r.version is ResponseVersion.V2) == 4


def test_run_improvement_judges_same_indices_for_v1_and_v2(monkeypatch, tmp_path: Path) -> None:
    """The same sampled tasks are judged for both versions (fair comparison)."""
    monkeypatch.setattr(
        runner_module,
        "judge_response",
        lambda *a, **k: [CriterionScore(name="correctness", score=0.5),
                         CriterionScore(name="completeness", score=0.5),
                         CriterionScore(name="clarity", score=0.5)],
    )
    dbp = tmp_path / "imp2.db"
    db.init_db(dbp)
    tasks = [Task(prompt=f"q{i} retrieval", source="golden") for i in range(10)]

    imp = run_improvement(
        tasks, _loop_runner, settings=_settings(judge_sample_rate=0.2), db_path=dbp
    )

    assert imp.n_judged == 2  # round(10 * 0.2)
    # For every task, v1 and v2 share the same judged flag (same sampled indices).
    for v1r, v2r in zip(imp.v1_results, imp.v2_results, strict=True):
        assert v1r.judged == v2r.judged
