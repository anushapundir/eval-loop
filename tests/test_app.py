"""Tests for the FastAPI application layer (app/main.py).

The API is a thin boundary over the existing agent + storage + reporting layers,
so these tests exercise the *wiring* with everything underneath faked: the graph
is monkeypatched to a scripted state (no Ollama) and the DB calls are stubbed to
no-ops / canned reads (no disk). They prove the endpoints assemble the right
payloads, not the agent/eval logic (covered by the runner/graph tests).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import app.main as app_main
from app.main import app
from storage.models import (
    AgentResponse,
    CriterionScore,
    EvalResult,
    Experiment,
    ResponseVersion,
    Task,
)


def _scripted_state(task: Task) -> dict:
    """A fully-populated AgentState dict mimicking a finished loop (v1 != v2)."""
    from agents.state import AgentState

    v1 = AgentResponse(task_id=task.id, version=ResponseVersion.V1, text="weak v1 answer",
                       retrieved_doc_ids=["rag"])
    v2 = AgentResponse(task_id=task.id, version=ResponseVersion.V2, text="improved v2 answer",
                       retrieved_doc_ids=["rag"])
    v1_eval = EvalResult(task_id=task.id, response_id=v1.id, version=ResponseVersion.V1,
                         deterministic=[CriterionScore(name="grounding", score=0.5)],
                         overall_score=0.5, passed=False)
    v2_eval = EvalResult(task_id=task.id, response_id=v2.id, version=ResponseVersion.V2,
                         deterministic=[CriterionScore(name="grounding", score=0.9)],
                         overall_score=0.9, passed=True)
    return AgentState(
        task=task, context="rag context", retrieved_doc_ids=["rag"],
        v1=v1, v2=v2, v1_eval=v1_eval, v2_eval=v2_eval, feedback="add grounding",
    ).model_dump()


class _FakeGraph:
    """Stand-in for a compiled LangGraph: .invoke returns a scripted state dict."""

    def invoke(self, state):
        return _scripted_state(state.task)


def _stub_db_writes(monkeypatch) -> None:
    """Neutralize every DB write/init so the API never touches disk in tests."""
    for name in ("init_db", "write_task", "write_response", "write_trace",
                 "write_eval_result", "write_experiment"):
        monkeypatch.setattr(app_main.db, name, lambda *a, **k: None)


def test_health_ok() -> None:
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_run_returns_v1_v2_and_delta(monkeypatch) -> None:
    monkeypatch.setattr(app_main, "build_graph", lambda: _FakeGraph())
    _stub_db_writes(monkeypatch)

    client = TestClient(app)
    resp = client.post("/run", json={"prompt": "What is LLM-as-judge?"})
    assert resp.status_code == 200
    body = resp.json()

    assert body["v1"]["text"] == "weak v1 answer"
    assert body["v2"]["text"] == "improved v2 answer"
    assert body["revised"] is True
    assert body["feedback"] == "add grounding"
    assert body["retrieved_doc_ids"] == ["rag"]
    # delta = mean v2 overall - mean v1 overall = 0.9 - 0.5
    assert round(body["improvement_delta"], 3) == 0.4
    assert body["v1_eval"]["overall_score"] == 0.5
    assert body["v2_eval"]["overall_score"] == 0.9


def test_run_rejects_blank_prompt() -> None:
    client = TestClient(app)
    resp = client.post("/run", json={"prompt": "   "})
    assert resp.status_code == 422


def test_run_returns_503_when_agent_fails(monkeypatch) -> None:
    class _BrokenGraph:
        def invoke(self, state):
            raise RuntimeError("ollama down")

    monkeypatch.setattr(app_main, "build_graph", lambda: _BrokenGraph())
    _stub_db_writes(monkeypatch)

    client = TestClient(app)
    resp = client.post("/run", json={"prompt": "anything"})
    assert resp.status_code == 503


def test_results_returns_experiment_summaries(monkeypatch) -> None:
    exp = Experiment(name="loop-golden", n_tasks=2, n_judged=0,
                     mean_v1=0.5, mean_v2=0.9, improvement_delta=0.4)
    v1 = EvalResult(task_id="t1", response_id="r1", version=ResponseVersion.V1,
                    deterministic=[CriterionScore(name="grounding", score=0.5)],
                    overall_score=0.5, passed=False, experiment_id=exp.id)
    v2 = EvalResult(task_id="t1", response_id="r2", version=ResponseVersion.V2,
                    deterministic=[CriterionScore(name="grounding", score=0.9)],
                    overall_score=0.9, passed=True, experiment_id=exp.id)

    monkeypatch.setattr(app_main.db, "init_db", lambda *a, **k: None)
    monkeypatch.setattr(app_main.db, "list_experiments", lambda *a, **k: [exp])
    monkeypatch.setattr(app_main.db, "list_eval_results", lambda *a, **k: [v1, v2])

    client = TestClient(app)
    resp = client.get("/results")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    summary = body[0]
    assert summary["experiment_id"] == exp.id
    assert summary["name"] == "loop-golden"
    assert summary["mean_v1"] == 0.5
    assert summary["mean_v2"] == 0.9
    assert summary["improvement_delta"] == 0.4
