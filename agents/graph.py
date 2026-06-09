"""LangGraph agent graph.

Day 2: a two-node pipeline — ``retrieve`` fetches grounding context from the
local KB, then ``generate`` produces the initial (v1) response from that
context. Each node emits a trace. Later days extend this into
``generate -> evaluate -> feedback -> revise -> re-evaluate`` with a
max-iterations stop (CLAUDE.md §8).
"""

from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from agents.llm import generate
from agents.prompts import GENERATION_SYSTEM, build_generation_prompt
from agents.state import AgentState
from agents.tools import load_kb, retrieve
from config.logging import get_logger
from config.settings import get_settings
from storage.models import AgentResponse, ResponseVersion, Trace

log = get_logger(__name__)


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
    completion = generate(prompt, system=GENERATION_SYSTEM)
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


def build_graph():
    """Compile and return the Day 2 agent graph (retrieve -> generate)."""
    builder = StateGraph(AgentState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)
    return builder.compile()
