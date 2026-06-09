"""Agent prompts — versioned, so improvement and experiments can swap them.

Prompts live here (not inline in the graph) for two reasons: the self-improvement
work edits prompts, and Day 5 experiments compare prompt versions. Each prompt
carries an explicit version string that flows into ``Experiment.prompt_version``.

The generation prompt instructs the model to answer *from the retrieved context*
and to admit when the context is insufficient — this is what makes the Day 3
grounding check meaningful and curbs hallucination.
"""

from __future__ import annotations

GENERATION_PROMPT_VERSION = "v1"

GENERATION_SYSTEM = (
    "You are a careful, concise assistant specializing in AI evaluation and "
    "agent engineering. Answer using ONLY the provided context. If the context "
    "does not contain the answer, say you don't have enough information rather "
    "than guessing. Do not invent facts or cite sources that are not in the "
    "context."
)

GENERATION_TEMPLATE = """Use ONLY the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""


def build_generation_prompt(*, question: str, context: str) -> str:
    """Render the generation prompt for a task and its retrieved context.

    Args:
        question: The user's task prompt.
        context: Retrieved KB text (may be empty if nothing matched).

    Returns:
        The full user prompt to send to the model.
    """
    return GENERATION_TEMPLATE.format(
        context=context if context.strip() else "(no relevant context found)",
        question=question,
    )
