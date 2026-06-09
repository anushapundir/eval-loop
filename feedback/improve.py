"""Revision step — rewrite a response using reviewer feedback (closes the loop).

This is the "doer" side of the self-improvement loop: given the prior answer,
the retrieved context, and structured feedback, it asks the agent model to
produce an improved v2. It runs on the default (free) provider — never the paid
judge — so the loop stays cost-safe (CLAUDE.md §2). The revision prompt takes
only feedback + context, so the golden reference answer can never leak in.
"""

from __future__ import annotations

from agents.llm import generate
from agents.prompts import REVISION_SYSTEM, build_revision_prompt
from config.settings import Settings


def revise(
    *,
    question: str,
    context: str,
    previous_answer: str,
    feedback: str,
    settings: Settings | None = None,
) -> str:
    """Produce a revised answer (v2) from feedback, grounded in the same context.

    Args:
        question: The task prompt.
        context: Retrieved KB context the answer must stay grounded in.
        previous_answer: The v1 response being improved.
        feedback: Rendered reviewer feedback to apply.
        settings: Injectable settings (defaults to the cached singleton); the
            provider is left to settings so revision uses the free model, not
            the judge.

    Returns:
        The revised response text.
    """
    prompt = build_revision_prompt(
        question=question,
        context=context,
        previous_answer=previous_answer,
        feedback=feedback,
    )
    completion = generate(prompt, system=REVISION_SYSTEM, settings=settings)
    return completion.text
