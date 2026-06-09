"""The judging rubric — scoring criteria as data, not logic (CLAUDE.md §7).

The rubric defines the *subjective* criteria the LLM judge scores: quality that
deterministic checks cannot capture. Anything with a structural or ground-truth
answer (grounding, coverage, length) stays in ``checks.py`` because it is free
and reliable; the judge is reserved for correctness, completeness, and clarity.

Each criterion carries concrete high/low anchors so the judge applies a
consistent standard (good rubric design — see datasets/kb/llm-as-judge.md). The
rubric is versioned so Day 5 experiments can attribute score changes to rubric
edits, mirroring ``GENERATION_PROMPT_VERSION``.
"""

from __future__ import annotations

from dataclasses import dataclass

RUBRIC_VERSION = "v1"


@dataclass(frozen=True)
class Criterion:
    """One subjective scoring criterion with concrete anchors."""

    name: str  # stable key; becomes CriterionScore.name and a JSON key
    description: str  # what the criterion measures
    high: str  # what a score near 1.0 looks like
    low: str  # what a score near 0.0 looks like


# Few and independent, so per-criterion scores stay interpretable.
RUBRIC: tuple[Criterion, ...] = (
    Criterion(
        name="correctness",
        description="Is the answer factually right given the task and provided context?",
        high="Every claim is accurate and consistent with the context.",
        low="Contains false or unsupported claims, or contradicts the context.",
    ),
    Criterion(
        name="completeness",
        description="Does the answer address the whole question, not just part of it?",
        high="Covers all parts of the question with the key points needed to be useful.",
        low="Omits important parts of the question or essential points.",
    ),
    Criterion(
        name="clarity",
        description="Is the answer clear, well-organized, and easy to follow?",
        high="Concise and well structured; a reader understands it on first read.",
        low="Rambling, disorganized, or confusing.",
    ),
)

# Stable list of criterion names the judge must return a score for.
CRITERION_NAMES: tuple[str, ...] = tuple(c.name for c in RUBRIC)


def render_rubric() -> str:
    """Render the rubric as text for the judge's (cacheable) system prompt."""
    lines = []
    for c in RUBRIC:
        lines.append(f"- {c.name}: {c.description}")
        lines.append(f"    high (1.0): {c.high}")
        lines.append(f"    low (0.0): {c.low}")
    return "\n".join(lines)
