"""LangGraph agent graph.

Day 1 skeleton: a single ``generate`` node that turns a task into an initial
(v1) response via the local model and records a trace. Later days extend this
into ``generate -> evaluate -> feedback -> revise -> re-evaluate`` with a
max-iterations stop (CLAUDE.md §8).
"""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from agents.llm import generate
from agents.state import AgentState
from config.logging import get_logger
from storage.models import AgentResponse, ResponseVersion, Trace

log = get_logger(__name__)

# Minimal generation prompt for the Day 1 slice; real prompts live in
# agents/prompts.py from Day 2 onward.
_SYSTEM = (
    "You are a careful, concise assistant. Answer the user's task directly and "
    "accurately. If you are unsure, say so rather than inventing facts."
)


def generate_node(state: AgentState) -> dict:
    """Produce the initial (v1) response for the task and emit a trace."""
    start = time.perf_counter()
    completion = generate(state.task.prompt, system=_SYSTEM)
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
        payload={"prompt": state.task.prompt, "chars": len(completion.text)},
    )
    log.info(
        "generate: task=%s provider=%s latency=%.0fms chars=%d",
        state.task.id, completion.provider, latency_ms, len(completion.text),
    )
    return {"v1": response, "traces": state.traces + [trace]}


def build_graph():
    """Compile and return the Day 1 agent graph (single generate node)."""
    builder = StateGraph(AgentState)
    builder.add_node("generate", generate_node)
    builder.set_entry_point("generate")
    builder.add_edge("generate", END)
    return builder.compile()
