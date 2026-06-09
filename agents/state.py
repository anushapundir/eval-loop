"""LangGraph state object — the data carried through the agent graph.

A single Pydantic model threads through every node: the task in, retrieval
context, the v1 and v2 responses, the evaluation/feedback between them, and the
iteration counter that bounds the loop. Nodes return partial updates to this
state.
"""

from __future__ import annotations

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

    iteration: int = 0
    max_iterations: int = 2

    # Traces accumulated during the run, for observability; the runner persists
    # these to storage after the graph completes.
    traces: list[Trace] = Field(default_factory=list)
