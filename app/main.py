"""Application layer — a thin FastAPI boundary over the agent (CLAUDE.md §4).

This is the integration seam between the UI and the system under test. The UI
never imports the agent directly; it talks to these endpoints, which keeps the
frontend and backend decoupled (and makes the loop callable over HTTP).

Endpoints:

* ``GET  /health``  — liveness check (no agent, no DB writes).
* ``POST /run``     — run the full critique→revise loop on one task and return
  v1, v2, both eval results, the feedback, and the improvement delta. Mirrors
  the proven ``cmd_run`` sequence in ``main.py``. Deterministic checks only by
  default (free); the paid Haiku judge is opt-in via ``do_judge`` (cost §2).
* ``GET  /results`` — list stored experiments as aggregated summaries (read-only;
  reused by the Streamlit results explorer).

The endpoints are deliberately thin: they assemble request/response Pydantic
models and delegate to ``agents.graph``, ``evaluators.runner``, ``storage.db``,
and ``reports.metrics``. Nothing here re-implements agent or eval logic.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from agents.graph import build_graph
from agents.state import AgentState
from config.logging import get_logger
from config.settings import get_settings
from evaluators.runner import evaluate_response
from reports.metrics import ExperimentSummary, summarize_experiment
from storage import db
from storage.models import AgentResponse, EvalResult, Task

log = get_logger("app")


# --- API request / response models (no raw dicts cross the boundary, §6) ----


class RunRequest(BaseModel):
    """A task submission to the agent loop."""

    prompt: str = Field(min_length=1, description="The user task/question.")
    do_judge: bool = Field(
        default=False,
        description="Also run the sampled Haiku judge on v1/v2 (costs API budget).",
    )
    provider: Literal["ollama", "haiku"] | None = Field(
        default=None,
        description="Per-run model override; None uses the configured default.",
    )
    key_points: list[str] = Field(
        default_factory=list,
        description="Required points for the coverage check; a missing point is "
        "what makes v1 fail and drives the revise loop (used by example tasks).",
    )
    pass_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Per-run quality bar; None uses the configured default. A "
        "stricter bar makes a borderline v1 fail so the loop revises.",
    )

    @field_validator("prompt")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        """Reject whitespace-only prompts (an empty task has nothing to ground)."""
        if not v.strip():
            raise ValueError("prompt must not be blank")
        return v


class TraceStep(BaseModel):
    """A compact view of one trace step for the UI timeline."""

    step: str
    provider: str
    latency_ms: float | None = None


class RunResponse(BaseModel):
    """The full result of one critique→revise loop, ready for the UI to render."""

    task: Task
    v1: AgentResponse
    v2: AgentResponse
    v1_eval: EvalResult
    v2_eval: EvalResult
    feedback: str | None = None
    revised: bool
    improvement_delta: float
    retrieved_doc_ids: list[str] = []
    judged: bool = False
    traces: list[TraceStep] = []


@asynccontextmanager
async def _lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Ensure runtime dirs + the SQLite schema exist before serving."""
    settings = get_settings()
    settings.ensure_dirs()
    db.init_db()
    log.info("API ready (provider=%s, db=%s).", settings.model_provider, settings.db_path)
    yield


app = FastAPI(
    title="eval-loop",
    summary="Self-improving agent with an evaluation-driven feedback loop.",
    lifespan=_lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check — no agent, no DB writes."""
    return {"status": "ok"}


@app.post("/run", response_model=RunResponse)
def run(request: RunRequest) -> RunResponse:
    """Run the agent loop on one task, persist it, and return v1 vs v2.

    Replicates ``cmd_run``: build a task, run the graph (retrieve → generate →
    evaluate → feedback → revise → re-evaluate), persist task/responses/traces/
    evals, and return a typed payload. When ``do_judge`` is set, v1 and v2 are
    re-scored with the paid judge *after* the loop (the judge never runs inside
    the loop — CLAUDE.md §2/§7).
    """
    settings = get_settings()
    task = Task(prompt=request.prompt, source="user", key_points=request.key_points)
    db.write_task(task)

    try:
        graph = build_graph()
        result = graph.invoke(AgentState(
            task=task,
            max_iterations=settings.max_iterations,
            provider=request.provider,
            pass_threshold=request.pass_threshold,
        ))
    except Exception as exc:  # noqa: BLE001 — surface any agent failure as 503, not 500.
        log.error("Agent run failed for task %s: %s", task.id, exc)
        raise HTTPException(
            status_code=503,
            detail="The agent runtime is unavailable (is Ollama running?).",
        ) from exc

    state = AgentState(**result)
    if state.v1 is None or state.v2 is None or state.v1_eval is None or state.v2_eval is None:
        log.error("Agent produced an incomplete loop for task %s.", task.id)
        raise HTTPException(status_code=500, detail="Agent produced an incomplete result.")

    v1_eval, v2_eval = state.v1_eval, state.v2_eval
    if request.do_judge:
        # Re-score both versions with the judge for a fair, judged comparison.
        v1_eval = evaluate_response(state.v1, task, state.context, do_judge=True, settings=settings)
        v2_eval = evaluate_response(state.v2, task, state.context, do_judge=True, settings=settings)

    db.write_response(state.v1)
    db.write_response(state.v2)
    for trace in state.traces:
        db.write_trace(trace)
    db.write_eval_result(v1_eval)
    db.write_eval_result(v2_eval)

    return RunResponse(
        task=task,
        v1=state.v1,
        v2=state.v2,
        v1_eval=v1_eval,
        v2_eval=v2_eval,
        feedback=state.feedback,
        revised=state.v2.text != state.v1.text,
        improvement_delta=round(v2_eval.overall_score - v1_eval.overall_score, 3),
        retrieved_doc_ids=state.v1.retrieved_doc_ids,
        judged=v1_eval.judged or v2_eval.judged,
        traces=[
            TraceStep(step=t.step, provider=t.provider, latency_ms=t.latency_ms)
            for t in state.traces
        ],
    )


def _sse(event: dict) -> str:
    """Format one event as a Server-Sent Events 'data:' frame."""
    return f"data: {json.dumps(event)}\n\n"


def _eval_payload(ev) -> dict:
    """Per-criterion deterministic scores + overall/passed for an EvalResult."""
    return {
        "scores": [{"name": s.name, "score": s.score} for s in ev.deterministic],
        "overall": ev.overall_score,
        "passed": ev.passed,
    }


def _judged_event(v1_eval: EvalResult, v2_eval: EvalResult) -> dict:
    """Build the SSE 'judged' event: the judge's per-criterion verdict, v1 vs v2.

    Pairs each rubric criterion's v1 and v2 score and carries the judge's written
    justification, so the UI can show *what* the judge concluded — not just the
    blended overall. Falls back gracefully if the judge degraded (empty lists).
    """
    v1_by_name = {s.name: s for s in v1_eval.judge}
    v2_by_name = {s.name: s for s in v2_eval.judge}
    criteria = [
        {
            "name": name,
            "v1": v1_by_name[name].score if name in v1_by_name else None,
            "v2": v2_by_name[name].score if name in v2_by_name else None,
            "v1_justification": (v1_by_name[name].justification if name in v1_by_name else None),
            "v2_justification": (v2_by_name[name].justification if name in v2_by_name else None),
        }
        for name in (list(v1_by_name) or list(v2_by_name))
    ]
    return {
        "type": "judged",
        "v1_overall": v1_eval.overall_score,
        "v2_overall": v2_eval.overall_score,
        "criteria": criteria,
    }


def _stage_event(node: str, delta: dict) -> dict:
    """Build the SSE stage event for one finished graph node.

    Reads provider/latency from the newest trace in the node's state delta and
    the intermediate output from the delta's state keys. Returns a JSON-safe dict.
    """
    traces = delta.get("traces") or []
    trace = traces[-1] if traces else None
    payload: dict = {}
    if node == "retrieve":
        payload = {"doc_ids": delta.get("retrieved_doc_ids") or []}
    elif node == "generate" and delta.get("v1") is not None:
        payload = {"text": delta["v1"].text}
    elif node == "evaluate_v1" and delta.get("v1_eval") is not None:
        payload = _eval_payload(delta["v1_eval"])
    elif node == "feedback":
        payload = {"text": delta.get("feedback") or ""}
    elif node == "revise" and delta.get("v2") is not None:
        payload = {"text": delta["v2"].text}
    elif node in ("evaluate_v2", "carry_forward") and delta.get("v2_eval") is not None:
        payload = _eval_payload(delta["v2_eval"])
        if node == "carry_forward":
            payload["reason"] = "v1 passed or revision budget exhausted"
    return {
        "type": "stage",
        "step": node,
        "provider": (trace.provider if trace else None),
        "latency_ms": (trace.latency_ms if trace else None),
        "payload": payload,
    }


@app.post("/run/stream")
def run_stream(request: RunRequest) -> StreamingResponse:
    """Run the loop and stream one SSE event per pipeline stage (real-time view).

    Mirrors ``/run`` but emits progress as it happens via ``graph.stream``. The
    paid judge still runs only *after* the loop when ``do_judge`` is set
    (CLAUDE.md §2/§7). Errors are delivered as in-band ``error`` events so the UI
    shows a clean message rather than a hung stream.
    """
    settings = get_settings()
    task = Task(prompt=request.prompt, source="user", key_points=request.key_points)

    def _generate():
        captured: dict = {}
        try:
            db.write_task(task)
            graph = build_graph()
            state = AgentState(
                task=task,
                max_iterations=settings.max_iterations,
                provider=request.provider,
                pass_threshold=request.pass_threshold,
            )
            for update in graph.stream(state, stream_mode="updates"):
                for node, delta in update.items():
                    captured.update(delta)
                    yield _sse(_stage_event(node, delta))
        except Exception as exc:  # noqa: BLE001 - surface as an in-band error event
            log.error("Streaming run failed for task %s: %s", task.id, exc)
            yield _sse({"type": "error",
                        "detail": "The agent runtime is unavailable (is Ollama running?)."})
            return

        v1 = captured.get("v1")
        v2 = captured.get("v2")
        v1_eval = captured.get("v1_eval")
        v2_eval = captured.get("v2_eval")
        context = captured.get("context", "")
        if v1 is None or v2 is None or v1_eval is None or v2_eval is None:
            yield _sse({"type": "error", "detail": "Agent produced an incomplete result."})
            return

        try:
            if request.do_judge:
                v1_eval = evaluate_response(v1, task, context, do_judge=True, settings=settings)
                v2_eval = evaluate_response(v2, task, context, do_judge=True, settings=settings)
                yield _sse(_judged_event(v1_eval, v2_eval))

            db.write_response(v1)
            db.write_response(v2)
            for trace in captured.get("traces", []):
                db.write_trace(trace)
            db.write_eval_result(v1_eval)
            db.write_eval_result(v2_eval)

            delta_score = round(v2_eval.overall_score - v1_eval.overall_score, 3)
            yield _sse({"type": "done",
                        "revised": v2.text != v1.text,
                        "improvement_delta": delta_score})
        except Exception as exc:  # noqa: BLE001 - surface post-loop failures as an in-band error
            log.error("Post-loop step failed for task %s: %s", task.id, exc)
            yield _sse({"type": "error",
                        "detail": "Post-processing failed (judge or storage error)."})

    return StreamingResponse(_generate(), media_type="text/event-stream")


@app.get("/results", response_model=list[ExperimentSummary])
def results() -> list[ExperimentSummary]:
    """List stored experiments as aggregated summaries (newest first, read-only)."""
    db.init_db()  # idempotent; tolerate a fresh deployment with no runs yet.
    experiments = db.list_experiments()
    return [
        summarize_experiment(exp, db.list_eval_results(exp.id)) for exp in experiments
    ]
