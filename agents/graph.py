"""LangGraph agent graph.

Day 4: the critique-and-revise loop. ``retrieve`` fetches grounding context,
``generate`` produces the initial (v1) response, ``evaluate_v1`` scores it with
the *free deterministic* checks, and a conditional then either revises (when v1
fails and the iteration budget allows) via ``feedback -> revise -> evaluate_v2``
or carries v1 forward unchanged. The paid LLM judge is never called inside the
loop (CLAUDE.md §2); it enters only at the dataset-comparison stage. Each node
emits a trace.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from langgraph.graph import END, StateGraph

from agents.llm import generate
from agents.prompts import GENERATION_SYSTEM, build_generation_prompt
from agents.state import AgentState
from agents.tools import load_kb, retrieve
from config.logging import get_logger
from config.settings import get_settings
from evaluators.checks import run_deterministic_checks
from feedback.generator import generate_feedback
from feedback.improve import revise
from storage.models import AgentResponse, EvalResult, ResponseVersion, Trace

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

log = get_logger(__name__)


def _deterministic_eval(response: AgentResponse, state: AgentState) -> EvalResult:
    """Score a response with the free deterministic suite (no judge in the loop)."""
    scores = run_deterministic_checks(
        response.text, context=state.context, key_points=state.task.key_points
    )
    overall = round(sum(s.score for s in scores) / len(scores), 3)
    return EvalResult(
        task_id=state.task.id,
        response_id=response.id,
        version=response.version,
        deterministic=scores,
        overall_score=overall,
        passed=overall >= get_settings().pass_threshold,
    )


def retrieve_node(state: AgentState) -> dict:
    """Fetch grounding context for the task from the local KB and trace it."""
    settings = get_settings()
    start = time.perf_counter()
    chunks = load_kb(settings.kb_dir)
    result = retrieve(state.task.prompt, chunks, k=settings.retrieval_top_k)
    latency_ms = (time.perf_counter() - start) * 1000.0

    trace = Trace(
        task_id=state.task.id,
        step="retrieve",
        latency_ms=latency_ms,
        payload={
            "query": state.task.prompt,
            "doc_ids": result.doc_ids,
            "n_chunks": len(result.chunks),
        },
    )
    log.info(
        "retrieve: task=%s docs=%s chunks=%d latency=%.0fms",
        state.task.id, result.doc_ids, len(result.chunks), latency_ms,
    )
    return {
        "retrieved_doc_ids": result.doc_ids,
        "context": result.context,
        "traces": state.traces + [trace],
    }


def generate_node(state: AgentState) -> dict:
    """Produce the initial (v1) response grounded in the retrieved context."""
    start = time.perf_counter()
    prompt = build_generation_prompt(question=state.task.prompt, context=state.context)
    completion = generate(prompt, system=GENERATION_SYSTEM, provider=state.provider)
    latency_ms = (time.perf_counter() - start) * 1000.0

    response = AgentResponse(
        task_id=state.task.id,
        version=ResponseVersion.V1,
        text=completion.text,
        retrieved_doc_ids=state.retrieved_doc_ids,
        model_provider=completion.provider,
    )
    trace = Trace(
        task_id=state.task.id,
        step="generate",
        response_id=response.id,
        provider=completion.provider,
        latency_ms=latency_ms,
        payload={"chars": len(completion.text), "doc_ids": state.retrieved_doc_ids},
    )
    log.info(
        "generate: task=%s provider=%s latency=%.0fms chars=%d",
        state.task.id, completion.provider, latency_ms, len(completion.text),
    )
    return {"v1": response, "traces": state.traces + [trace]}


def evaluate_v1_node(state: AgentState) -> dict:
    """Score v1 with the free deterministic checks; drives the revise decision."""
    ev = _deterministic_eval(state.v1, state)
    trace = Trace(
        task_id=state.task.id,
        step="evaluate",
        response_id=state.v1.id,
        payload={"version": "v1", "overall": ev.overall_score, "passed": ev.passed},
    )
    log.info("evaluate v1: task=%s overall=%.3f passed=%s",
             state.task.id, ev.overall_score, ev.passed)
    return {"v1_eval": ev, "traces": state.traces + [trace]}


def should_revise(state: AgentState) -> str:
    """Revise only when v1 failed and the iteration budget allows; else carry forward."""
    if state.v1_eval is not None and not state.v1_eval.passed \
            and state.iteration < state.max_iterations:
        return "feedback"
    return "carry_forward"


def feedback_node(state: AgentState) -> dict:
    """Turn v1's evaluation into structured, actionable feedback (free, rule-based)."""
    fb = generate_feedback(state.v1_eval, state.task, state.v1.text)
    trace = Trace(
        task_id=state.task.id,
        step="feedback",
        response_id=state.v1.id,
        payload={"items": [i.model_dump() for i in fb.items], "n_items": len(fb.items)},
    )
    log.info("feedback: task=%s items=%d", state.task.id, len(fb.items))
    return {
        "feedback": fb.text,
        "iteration": state.iteration + 1,
        "traces": state.traces + [trace],
    }


def revise_node(state: AgentState) -> dict:
    """Produce the revised (v2) response by applying the feedback to v1."""
    settings = get_settings()
    chosen_provider = state.provider or settings.model_provider
    start = time.perf_counter()
    text = revise(
        question=state.task.prompt,
        context=state.context,
        previous_answer=state.v1.text,
        feedback=state.feedback or "",
        provider=chosen_provider,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0

    response = AgentResponse(
        task_id=state.task.id,
        version=ResponseVersion.V2,
        text=text,
        retrieved_doc_ids=state.retrieved_doc_ids,
        model_provider=chosen_provider,
    )
    trace = Trace(
        task_id=state.task.id,
        step="revise",
        response_id=response.id,
        provider=chosen_provider,
        latency_ms=latency_ms,
        payload={"chars": len(text)},
    )
    log.info("revise: task=%s provider=%s latency=%.0fms chars=%d",
             state.task.id, chosen_provider, latency_ms, len(text))
    return {"v2": response, "traces": state.traces + [trace]}


def evaluate_v2_node(state: AgentState) -> dict:
    """Score the revised v2 with the same free deterministic checks."""
    ev = _deterministic_eval(state.v2, state)
    trace = Trace(
        task_id=state.task.id,
        step="evaluate",
        response_id=state.v2.id,
        payload={"version": "v2", "overall": ev.overall_score, "passed": ev.passed},
    )
    log.info("evaluate v2: task=%s overall=%.3f passed=%s",
             state.task.id, ev.overall_score, ev.passed)
    return {"v2_eval": ev, "traces": state.traces + [trace]}


def carry_forward_node(state: AgentState) -> dict:
    """When v1 already passes (or revision is disabled), v2 carries v1 forward.

    v2 is a distinct V2 response with v1's text, so the dataset comparison always
    has a v2 and the delta is an honest zero rather than a missing value.
    """
    response = AgentResponse(
        task_id=state.task.id,
        version=ResponseVersion.V2,
        text=state.v1.text,
        retrieved_doc_ids=state.retrieved_doc_ids,
        model_provider=state.v1.model_provider,
    )
    ev = _deterministic_eval(response, state)
    trace = Trace(
        task_id=state.task.id,
        step="carry_forward",
        response_id=response.id,
        payload={"reason": "v1 passed or revision budget exhausted"},
    )
    log.info("carry_forward: task=%s (no revision)", state.task.id)
    return {"v2": response, "v2_eval": ev, "traces": state.traces + [trace]}


def build_graph() -> CompiledStateGraph:
    """Compile the Day 4 critique-and-revise graph with a max-iterations stop."""
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("evaluate_v1", evaluate_v1_node)
    builder.add_node("feedback", feedback_node)
    builder.add_node("revise", revise_node)
    builder.add_node("evaluate_v2", evaluate_v2_node)
    builder.add_node("carry_forward", carry_forward_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", "evaluate_v1")
    builder.add_conditional_edges(
        "evaluate_v1",
        should_revise,
        {"feedback": "feedback", "carry_forward": "carry_forward"},
    )
    builder.add_edge("feedback", "revise")
    builder.add_edge("revise", "evaluate_v2")
    builder.add_edge("evaluate_v2", END)
    builder.add_edge("carry_forward", END)
    return builder.compile()
