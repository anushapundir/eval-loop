"""LangGraph state object — the data carried through the agent graph.

A single Pydantic model threads through every node: the task in, retrieval
context, the v1 and v2 responses, the evaluation/feedback between them, and the
iteration counter that bounds the loop. Nodes return partial updates to this
state.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from storage.models import AgentResponse, EvalResult, Task, Trace


class AgentState(BaseModel):
    """Mutable state for one pass through the critique-and-revise graph."""

    task: Task
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    context: str = Field(default="", description="Concatenated retrieved KB text.")

    v1: AgentResponse | None = None
    v2: AgentResponse | None = None

    v1_eval: EvalResult | None = None
    v2_eval: EvalResult | None = None
    feedback: str | None = None

    # Optional per-run model override (e.g. "ollama" or "haiku"); None → use the
    # configured default provider. Threaded into the generate/revise model calls.
    provider: Literal["ollama", "haiku"] | None = None

    # Optional per-run quality bar; None → use settings.pass_threshold. A stricter
    # bar makes a borderline v1 fail, which is what drives the revise loop to
    # engage (otherwise good answers always pass and carry forward unchanged).
    pass_threshold: float | None = None

    iteration: int = 0
    max_iterations: int = 2

    # Traces accumulated during the run, for observability; the runner persists
    # these to storage after the graph completes.
    traces: list[Trace] = Field(default_factory=list)
