"""Pydantic v2 data models — the schema every layer depends on.

These models are the system's integration boundary (CLAUDE.md §4.1): tasks in,
responses out, eval results, and experiment records all cross layers as
validated objects, never as raw dicts. The SQLite schema in ``db.py`` mirrors
these.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    """UTC timestamp for record creation."""
    return datetime.now(UTC)


def _new_id() -> str:
    """Short unique id for a record."""
    return uuid4().hex


class ResponseVersion(StrEnum):
    """Which pass of the loop produced a response."""

    V1 = "v1"  # initial response
    V2 = "v2"  # revised response after feedback


class Task(BaseModel):
    """A unit of work given to the agent."""

    id: str = Field(default_factory=_new_id)
    prompt: str
    source: str = Field(default="user", description="user | golden | synthetic")
    expected: str | None = Field(
        default=None, description="Reference answer/label (golden only)."
    )
    key_points: list[str] = Field(
        default_factory=list, description="Required points for coverage checks."
    )
    created_at: datetime = Field(default_factory=_now)


class AgentResponse(BaseModel):
    """A response the agent produced for a task, with its retrieval context."""

    id: str = Field(default_factory=_new_id)
    task_id: str
    version: ResponseVersion = Field(default=ResponseVersion.V1)
    text: str
    retrieved_doc_ids: list[str] = Field(default_factory=list)
    model_provider: str = Field(default="ollama")
    created_at: datetime = Field(default_factory=_now)


class Trace(BaseModel):
    """An observability record of one agent execution step.

    One row per meaningful step (e.g. retrieval, generate, revise) so a run can
    be reconstructed and audited offline.
    """

    id: str = Field(default_factory=_new_id)
    task_id: str
    step: str = Field(description="e.g. 'generate', 'retrieve', 'revise'.")
    response_id: str | None = None
    provider: str = Field(default="ollama")
    latency_ms: float | None = None
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Step-specific details (inputs/outputs)."
    )
    created_at: datetime = Field(default_factory=_now)


class CriterionScore(BaseModel):
    """A single named score with an optional justification."""

    name: str
    score: float = Field(ge=0.0, le=1.0)
    justification: str | None = None


class EvalResult(BaseModel):
    """The structured score for one response (deterministic + optional judge)."""

    id: str = Field(default_factory=_new_id)
    task_id: str
    response_id: str
    version: ResponseVersion
    deterministic: list[CriterionScore] = Field(default_factory=list)
    judge: list[CriterionScore] = Field(default_factory=list)
    overall_score: float = Field(ge=0.0, le=1.0)
    passed: bool = False
    judged: bool = Field(default=False, description="True if the LLM judge ran.")
    experiment_id: str | None = None
    created_at: datetime = Field(default_factory=_now)


class Experiment(BaseModel):
    """A tracked run over the dataset; the unit of comparison across iterations."""

    id: str = Field(default_factory=_new_id)
    name: str
    prompt_version: str = Field(default="v1")
    model_provider: str = Field(default="ollama")
    n_tasks: int = 0
    n_judged: int = 0
    mean_v1: float | None = None
    mean_v2: float | None = None
    improvement_delta: float | None = None
    notes: str | None = None
    created_at: datetime = Field(default_factory=_now)
