"""Round-trip tests for the SQLite storage layer (uses a temp DB, no model calls)."""

from __future__ import annotations

from pathlib import Path

from storage import db
from storage.models import (
    AgentResponse,
    CriterionScore,
    EvalResult,
    Experiment,
    ResponseVersion,
    Task,
    Trace,
)


def test_trace_and_response_roundtrip(tmp_path: Path) -> None:
    dbp = tmp_path / "t.db"
    db.init_db(dbp)

    task = Task(prompt="hi", source="user")
    db.write_task(task, dbp)

    resp = AgentResponse(task_id=task.id, text="hello world", retrieved_doc_ids=["a", "b"])
    db.write_response(resp, dbp)

    trace = Trace(task_id=task.id, step="generate", response_id=resp.id,
                  latency_ms=12.5, payload={"chars": 11})
    db.write_trace(trace, dbp)

    fetched = db.get_trace(trace.id, dbp)
    assert fetched is not None
    assert fetched.payload == {"chars": 11}
    assert fetched.latency_ms == 12.5

    fetched_resp = db.get_response(resp.id, dbp)
    assert fetched_resp is not None
    assert fetched_resp.retrieved_doc_ids == ["a", "b"]
    assert fetched_resp.version is ResponseVersion.V1

    assert len(db.list_traces(task.id, dbp)) == 1


def test_eval_and_experiment_roundtrip(tmp_path: Path) -> None:
    dbp = tmp_path / "e.db"
    db.init_db(dbp)

    exp = Experiment(name="baseline", n_tasks=3)
    db.write_experiment(exp, dbp)

    result = EvalResult(
        task_id="t1",
        response_id="r1",
        version=ResponseVersion.V1,
        deterministic=[CriterionScore(name="length", score=1.0, justification=None)],
        judge=[CriterionScore(name="clarity", score=0.8, justification="clear")],
        overall_score=0.9,
        passed=True,
        judged=True,
        experiment_id=exp.id,
    )
    db.write_eval_result(result, dbp)

    results = db.list_eval_results(exp.id, dbp)
    assert len(results) == 1
    assert results[0].judge[0].name == "clarity"
    assert results[0].passed is True

    experiments = db.list_experiments(dbp)
    assert experiments[0].name == "baseline"
