"""Tests for the FastAPI application layer (app/main.py).

The API is a thin boundary over the existing agent + storage + reporting layers,
so these tests exercise the *wiring* with everything underneath faked: the graph
is monkeypatched to a scripted state (no Ollama) and the DB calls are stubbed to
no-ops / canned reads (no disk). They prove the endpoints assemble the right
payloads, not the agent/eval logic (covered by the runner/graph tests).
"""

from __future__ import annotations

import json

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
    Trace,
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


def test_run_passes_key_points_and_threshold_into_state(monkeypatch) -> None:
    """key_points and pass_threshold from the request reach the Task and AgentState."""
    captured = {}

    class _CapturingGraph:
        def invoke(self, state):
            captured["state"] = state
            return _scripted_state(state.task)

    monkeypatch.setattr(app_main, "build_graph", lambda: _CapturingGraph())
    _stub_db_writes(monkeypatch)

    client = TestClient(app)
    resp = client.post("/run", json={
        "prompt": "Why score each criterion separately?",
        "key_points": ["where", "weak", "overall score"],
        "pass_threshold": 0.9,
    })
    assert resp.status_code == 200
    state = captured["state"]
    assert state.task.key_points == ["where", "weak", "overall score"]
    assert state.pass_threshold == 0.9


def test_run_rejects_out_of_range_threshold() -> None:
    client = TestClient(app)
    resp = client.post("/run", json={"prompt": "q", "pass_threshold": 1.5})
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


# ---------------------------------------------------------------------------
# POST /run/stream — SSE streaming endpoint
# ---------------------------------------------------------------------------


def _trace(task_id, step, provider="ollama", latency=5.0, **payload):
    return Trace(task_id=task_id, step=step, provider=provider,
                 latency_ms=latency, payload=payload)


def _revise_updates(task):
    """Scripted graph.stream() updates for a full revise path (v1 fails → v2)."""
    from storage.models import AgentResponse, CriterionScore, EvalResult, ResponseVersion

    v1 = AgentResponse(task_id=task.id, version=ResponseVersion.V1, text="weak v1",
                       retrieved_doc_ids=["rag"])
    v2 = AgentResponse(task_id=task.id, version=ResponseVersion.V2, text="strong v2",
                       retrieved_doc_ids=["rag"])
    v1_eval = EvalResult(task_id=task.id, response_id=v1.id, version=ResponseVersion.V1,
                         deterministic=[CriterionScore(name="grounding", score=0.5)],
                         overall_score=0.5, passed=False)
    v2_eval = EvalResult(task_id=task.id, response_id=v2.id, version=ResponseVersion.V2,
                         deterministic=[CriterionScore(name="grounding", score=0.9)],
                         overall_score=0.9, passed=True)
    return [
        {"retrieve": {"retrieved_doc_ids": ["rag"], "context": "ctx",
                      "traces": [_trace(task.id, "retrieve", doc_ids=["rag"], n_chunks=2)]}},
        {"generate": {"v1": v1, "traces": [_trace(task.id, "generate")]}},
        {"evaluate_v1": {"v1_eval": v1_eval, "traces": [_trace(task.id, "evaluate")]}},
        {"feedback": {"feedback": "add grounding", "iteration": 1,
                      "traces": [_trace(task.id, "feedback")]}},
        {"revise": {"v2": v2, "traces": [_trace(task.id, "revise")]}},
        {"evaluate_v2": {"v2_eval": v2_eval, "traces": [_trace(task.id, "evaluate")]}},
    ]


class _FakeStreamGraph:
    def __init__(self, updates_for):
        self._updates_for = updates_for

    def stream(self, state, stream_mode=None):
        yield from self._updates_for(state.task)


def _events(resp_text):
    """Parse SSE 'data: {...}' lines into a list of event dicts."""
    return [json.loads(line[len("data: "):]) for line in resp_text.splitlines()
            if line.startswith("data: ")]


def test_run_stream_emits_stage_events_in_order(monkeypatch):
    monkeypatch.setattr(app_main, "build_graph", lambda: _FakeStreamGraph(_revise_updates))
    _stub_db_writes(monkeypatch)

    client = TestClient(app)
    resp = client.post("/run/stream", json={"prompt": "What is LLM-as-judge?"})
    assert resp.status_code == 200
    events = _events(resp.text)

    steps = [e["step"] for e in events if e["type"] == "stage"]
    assert steps == ["retrieve", "generate", "evaluate_v1", "feedback", "revise", "evaluate_v2"]

    gen = next(e for e in events if e.get("step") == "generate")
    assert gen["provider"] == "ollama"
    assert gen["payload"]["text"] == "weak v1"

    done = events[-1]
    assert done["type"] == "done"
    assert done["revised"] is True
    assert round(done["improvement_delta"], 3) == 0.4


def test_run_stream_carry_forward_path(monkeypatch):
    def _carry_updates(task):
        from storage.models import AgentResponse, CriterionScore, EvalResult, ResponseVersion
        v1 = AgentResponse(task_id=task.id, version=ResponseVersion.V1, text="good v1",
                           retrieved_doc_ids=["rag"])
        v2 = AgentResponse(task_id=task.id, version=ResponseVersion.V2, text="good v1",
                           retrieved_doc_ids=["rag"])
        ev1 = EvalResult(task_id=task.id, response_id=v1.id, version=ResponseVersion.V1,
                         deterministic=[CriterionScore(name="grounding", score=0.9)],
                         overall_score=0.9, passed=True)
        ev2 = EvalResult(task_id=task.id, response_id=v2.id, version=ResponseVersion.V2,
                         deterministic=[CriterionScore(name="grounding", score=0.9)],
                         overall_score=0.9, passed=True)
        return [
            {"retrieve": {"retrieved_doc_ids": ["rag"], "context": "ctx",
                          "traces": [_trace(task.id, "retrieve")]}},
            {"generate": {"v1": v1, "traces": [_trace(task.id, "generate")]}},
            {"evaluate_v1": {"v1_eval": ev1, "traces": [_trace(task.id, "evaluate")]}},
            {"carry_forward": {"v2": v2, "v2_eval": ev2,
                               "traces": [_trace(task.id, "carry_forward")]}},
        ]

    monkeypatch.setattr(app_main, "build_graph", lambda: _FakeStreamGraph(_carry_updates))
    _stub_db_writes(monkeypatch)

    client = TestClient(app)
    resp = client.post("/run/stream", json={"prompt": "q"})
    events = _events(resp.text)
    steps = [e["step"] for e in events if e["type"] == "stage"]
    assert steps == ["retrieve", "generate", "evaluate_v1", "carry_forward"]
    assert "feedback" not in steps and "revise" not in steps
    assert events[-1]["type"] == "done"
    assert events[-1]["revised"] is False


def test_run_stream_error_event_when_graph_fails(monkeypatch):
    class _BrokenStreamGraph:
        def stream(self, state, stream_mode=None):
            raise RuntimeError("ollama down")
            yield  # pragma: no cover - makes this a generator

    monkeypatch.setattr(app_main, "build_graph", lambda: _BrokenStreamGraph())
    _stub_db_writes(monkeypatch)

    client = TestClient(app)
    resp = client.post("/run/stream", json={"prompt": "q"})
    assert resp.status_code == 200
    events = _events(resp.text)
    assert any(e["type"] == "error" for e in events)


def test_run_stream_judged_event_when_do_judge(monkeypatch):
    monkeypatch.setattr(app_main, "build_graph", lambda: _FakeStreamGraph(_revise_updates))
    _stub_db_writes(monkeypatch)

    def _fake_eval(response, task, context, *, do_judge, settings=None, experiment_id=None):
        from storage.models import CriterionScore, EvalResult
        score = 0.8 if response.version.value == "v1" else 0.95
        return EvalResult(task_id=task.id, response_id=response.id, version=response.version,
                          deterministic=[CriterionScore(name="grounding", score=score)],
                          judge=[CriterionScore(name="correctness", score=score,
                                                justification=f"{response.version.value} note")],
                          overall_score=score, passed=True, judged=True)

    monkeypatch.setattr(app_main, "evaluate_response", _fake_eval)

    client = TestClient(app)
    resp = client.post("/run/stream", json={"prompt": "q", "do_judge": True})
    events = _events(resp.text)
    judged = next(e for e in events if e["type"] == "judged")
    assert judged["v1_overall"] == 0.8
    assert judged["v2_overall"] == 0.95
    # The judge's per-criterion verdict (v1 vs v2 + justifications) is carried.
    crit = next(c for c in judged["criteria"] if c["name"] == "correctness")
    assert crit["v1"] == 0.8 and crit["v2"] == 0.95
    assert crit["v1_justification"] == "v1 note" and crit["v2_justification"] == "v2 note"
    done = next(e for e in events if e["type"] == "done")
    assert round(done["improvement_delta"], 3) == 0.15


def test_judged_event_pairs_criteria_with_justifications() -> None:
    """_judged_event pairs each criterion's v1/v2 score and carries justifications."""
    v1 = EvalResult(task_id="t", response_id="r1", version=ResponseVersion.V1,
                    deterministic=[CriterionScore(name="grounding", score=0.5)],
                    judge=[CriterionScore(name="correctness", score=0.6, justification="v1 shaky"),
                           CriterionScore(name="clarity", score=0.8, justification="v1 ok")],
                    overall_score=0.55, passed=False, judged=True)
    v2 = EvalResult(task_id="t", response_id="r2", version=ResponseVersion.V2,
                    deterministic=[CriterionScore(name="grounding", score=0.9)],
                    judge=[CriterionScore(name="correctness", score=0.9, justification="v2 right"),
                           CriterionScore(name="clarity", score=0.85, justification="v2 clear")],
                    overall_score=0.9, passed=True, judged=True)

    ev = app_main._judged_event(v1, v2)
    assert ev["type"] == "judged"
    assert ev["v1_overall"] == 0.55 and ev["v2_overall"] == 0.9
    by_name = {c["name"]: c for c in ev["criteria"]}
    assert set(by_name) == {"correctness", "clarity"}
    assert by_name["correctness"]["v1"] == 0.6 and by_name["correctness"]["v2"] == 0.9
    assert by_name["correctness"]["v1_justification"] == "v1 shaky"
    assert by_name["correctness"]["v2_justification"] == "v2 right"


def test_judged_event_handles_degraded_judge() -> None:
    """When the judge degraded (empty judge lists), criteria is empty, not an error."""
    v1 = EvalResult(task_id="t", response_id="r1", version=ResponseVersion.V1,
                    deterministic=[CriterionScore(name="grounding", score=0.5)],
                    overall_score=0.5, passed=False, judged=False)
    v2 = EvalResult(task_id="t", response_id="r2", version=ResponseVersion.V2,
                    deterministic=[CriterionScore(name="grounding", score=0.9)],
                    overall_score=0.9, passed=True, judged=False)

    ev = app_main._judged_event(v1, v2)
    assert ev["criteria"] == []
    assert ev["v1_overall"] == 0.5 and ev["v2_overall"] == 0.9


def test_run_stream_error_event_when_judge_fails(monkeypatch):
    monkeypatch.setattr(app_main, "build_graph", lambda: _FakeStreamGraph(_revise_updates))
    _stub_db_writes(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("haiku down")

    monkeypatch.setattr(app_main, "evaluate_response", _boom)

    client = TestClient(app)
    resp = client.post("/run/stream", json={"prompt": "q", "do_judge": True})
    assert resp.status_code == 200
    events = _events(resp.text)
    assert any(e["type"] == "error" for e in events)
    assert not any(e["type"] == "done" for e in events)


def test_run_stream_rejects_blank_prompt():
    client = TestClient(app)
    resp = client.post("/run/stream", json={"prompt": "   "})
    assert resp.status_code == 422
