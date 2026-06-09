"""Synthetic dataset generation — the spec, and a deterministic generator.

Per CLAUDE.md §3 (Day 3), the synthetic set is *generated once and frozen*: run
this module to (re)write ``datasets/synthetic/synthetic.jsonl``, then commit the
file and treat it as fixed. Evaluation always reads the committed file, never
regenerates at runtime.

Honest framing: these are **template-generated** inputs, not LLM-generated. We
combine a fixed grid of question templates (varying tone, length and phrasing)
with topics drawn from the five KB documents. This keeps generation free and
fully deterministic (reproducible), which is what we want for a fixed eval set.
Synthetic tasks are deliberately *unlabeled* (no ``expected``/``key_points``):
they exercise volume and measure score *trend*, while the human-verified golden
set carries the ground truth for correctness and judge validation.

Generation spec (the "prompt" we would otherwise hand to an LLM):
    "Produce short user questions about AI evaluation and agent engineering,
     one per concept in the knowledge base, varying tone from terse to detailed
     and including casual/confused phrasings. No answers, just questions."
"""

from __future__ import annotations

from pathlib import Path

from config.settings import get_settings
from storage.models import Task

# Question templates, ordered from terse to detailed to casual. Iterating
# template-major over the topic list (see ``build_synthetic``) guarantees every
# topic appears under several templates within the first ~100 records.
_TEMPLATES: tuple[str, ...] = (
    "What is {topic}?",
    "Explain {topic} in simple terms.",
    "Briefly, how does {topic} work?",
    "In detail, explain {topic} and why it matters.",
    "I'm a bit confused about {topic} - can you clarify?",
    "Give me a concise definition of {topic}.",
)

# Topics drawn from the five KB docs (RAG, LLM-as-judge, evaluation metrics,
# feedback loops, agent design). Lowercase noun phrases that slot into templates.
_TOPICS: tuple[str, ...] = (
    "retrieval-augmented generation",
    "grounding in a RAG system",
    "when to use RAG",
    "keyword-based retrieval",
    "embedding-based retrieval",
    "the LLM-as-a-judge approach",
    "position bias in LLM judges",
    "length bias in LLM judges",
    "validating an LLM judge against a golden set",
    "rubric design for judging",
    "controlling the cost of LLM judging",
    "deterministic evaluation checks",
    "criteria-based evaluation",
    "per-criterion scoring",
    "the pass rate metric",
    "the improvement delta",
    "sample-size honesty in evaluation",
    "the critique-and-revise loop",
    "self-refinement (Reflexion)",
    "stop criteria for feedback loops",
    "avoiding overfitting in feedback loops",
    "the agent loop",
    "tool use in agents",
    "agent state",
    "orchestration graphs",
    "observability and tracing in agents",
)

_DEFAULT_LIMIT = 100


def build_synthetic(limit: int = _DEFAULT_LIMIT) -> list[Task]:
    """Build the frozen synthetic task list deterministically.

    Iterates the template grid template-major (every topic under template 0,
    then template 1, ...) so coverage stays broad as ``limit`` grows. Prompts
    are distinct by construction. Tasks are unlabeled (``source="synthetic"``,
    no ``expected``/``key_points``).

    Args:
        limit: Maximum number of tasks to produce.

    Returns:
        Up to ``limit`` synthetic ``Task`` objects, in a stable order.
    """
    tasks: list[Task] = []
    for template in _TEMPLATES:
        for topic in _TOPICS:
            if len(tasks) >= limit:
                return tasks
            tasks.append(Task(prompt=template.format(topic=topic), source="synthetic"))
    return tasks


def write_synthetic(path: Path | None = None, limit: int = _DEFAULT_LIMIT) -> Path:
    """Write the synthetic dataset to a JSONL file and return its path.

    Only the prompt-bearing fields are serialized (ids/timestamps are produced
    fresh at load time), so the frozen file is stable across regenerations.
    """
    settings = get_settings()
    out = path or (settings.synthetic_dir / "synthetic.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f'{{"prompt": {_json_str(t.prompt)}, "source": "synthetic"}}'
        for t in build_synthetic(limit)
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def _json_str(value: str) -> str:
    """Render a string as a JSON string literal (escapes quotes/backslashes)."""
    import json

    return json.dumps(value)


if __name__ == "__main__":
    written = write_synthetic()
    print(f"Wrote synthetic dataset: {written}")
