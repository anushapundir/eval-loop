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

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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
    title="eval-loop-agent",
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
    task = Task(prompt=request.prompt, source="user")
    db.write_task(task)

    try:
        graph = build_graph()
        result = graph.invoke(AgentState(task=task, max_iterations=settings.max_iterations))
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


@app.get("/results", response_model=list[ExperimentSummary])
def results() -> list[ExperimentSummary]:
    """List stored experiments as aggregated summaries (newest first, read-only)."""
    db.init_db()  # idempotent; tolerate a fresh deployment with no runs yet.
    experiments = db.list_experiments()
    return [
        summarize_experiment(exp, db.list_eval_results(exp.id)) for exp in experiments
    ]
