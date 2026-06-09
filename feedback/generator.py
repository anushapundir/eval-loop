"""Feedback generator — turns an EvalResult into structured, actionable critique.

This is the "critic" role of the loop, kept distinct from the doer (the agent)
and the judge (the evaluator) — CLAUDE.md §5. It is deliberately *deterministic
and free*: it maps low deterministic scores to concrete instructions rather than
calling a model. The most useful signal is coverage — it names the exact key
points the answer is missing, which gives the revision step something specific to
fix. It uses the task's ``key_points`` (a structural spec, available at inference)
but never the golden ``expected`` answer, so the loop's improvement is real.
"""

from __future__ import annotations

from agents.tools import tokenize
from config.settings import Settings, get_settings
from storage.models import EvalResult, Feedback, FeedbackItem, Task


def _score(result: EvalResult, name: str) -> float | None:
    """Return the deterministic score named ``name``, or None if absent."""
    for s in result.deterministic:
        if s.name == name:
            return s.score
    return None


def _missing_key_points(key_points: list[str], response_text: str) -> list[str]:
    """Key points whose content tokens are not all present in the response.

    Uses the same tokenizer as the coverage check (one tokenizer, no drift), so
    "missing" here means exactly what the coverage score measured.
    """
    response_tokens = tokenize(response_text)
    missing: list[str] = []
    for kp in key_points:
        kp_tokens = tokenize(kp)
        if kp_tokens and not kp_tokens <= response_tokens:
            missing.append(kp)
    return missing


def generate_feedback(
    result: EvalResult,
    task: Task,
    response_text: str,
    *,
    settings: Settings | None = None,
) -> Feedback:
    """Build structured feedback from a response's deterministic scores.

    Args:
        result: The deterministic evaluation of the response.
        task: The task (supplies ``key_points`` for coverage feedback).
        response_text: The response being critiqued (to locate missing points).
        settings: Injectable settings (for ``pass_threshold``).

    Returns:
        A :class:`Feedback`; ``is_actionable`` is False when nothing needs fixing.
    """
    settings = settings or get_settings()
    threshold = settings.pass_threshold
    items: list[FeedbackItem] = []

    non_empty = _score(result, "non_empty")
    if non_empty is not None and non_empty < 1.0:
        items.append(
            FeedbackItem(
                criterion="non_empty",
                problem="The answer is empty.",
                suggestion="Provide an actual answer grounded in the context.",
            )
        )

    coverage = _score(result, "coverage")
    if coverage is not None and coverage < 1.0 and task.key_points:
        missing = _missing_key_points(task.key_points, response_text)
        if missing:
            items.append(
                FeedbackItem(
                    criterion="coverage",
                    problem=f"The answer is missing {len(missing)} required point(s).",
                    suggestion="Make sure to address: " + ", ".join(missing) + ".",
                )
            )

    grounding = _score(result, "grounding")
    if grounding is not None and grounding < threshold:
        items.append(
            FeedbackItem(
                criterion="grounding",
                problem="Parts of the answer are not supported by the retrieved context.",
                suggestion=(
                    "Base every claim on the provided context; remove or correct anything "
                    "the context does not support."
                ),
            )
        )

    length = _score(result, "length")
    if length is not None and length < 1.0:
        why = next(
            (s.justification for s in result.deterministic if s.name == "length"), None
        )
        too_short = bool(why and "short" in why.lower())
        items.append(
            FeedbackItem(
                criterion="length",
                problem=why or "The answer length is outside the sensible range.",
                suggestion=(
                    "Expand the answer with relevant detail from the context."
                    if too_short
                    else "Make the answer more concise; keep only what the question needs."
                ),
            )
        )

    text = _render(items)
    return Feedback(items=items, text=text)


def _render(items: list[FeedbackItem]) -> str:
    """Render feedback items into the bullet list handed to the revision prompt."""
    if not items:
        return ""
    lines = ["Revise the previous answer to address the following:"]
    lines.extend(f"- {item.suggestion}" for item in items)
    return "\n".join(lines)
