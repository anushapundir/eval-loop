"""LLM-as-judge — sampled, cached Haiku scoring of subjective quality.

The judge is the only paid path in default (cost-safe) operation, so it is used
sparingly: the runner samples which responses reach it (CLAUDE.md §2). Here we
just score one response. The judge is always forced onto Haiku regardless of
``MODEL_PROVIDER``, with the rubric/system block cached across the batch.

Robustness (per the Day 3 plan): the rubric demands a strict JSON object; we
parse it, tolerating prose or code fences around it, retry a few times on
unparseable output, and on final failure return an empty list so the response
is simply scored deterministic-only (``judged=False``) — never crashing a batch.
"""

from __future__ import annotations

import json

from agents.llm import LLMError, generate
from agents.prompts import JUDGE_SYSTEM, build_judge_prompt
from config.logging import get_logger
from config.settings import Settings, get_settings
from evaluators.rubric import CRITERION_NAMES
from storage.models import CriterionScore

log = get_logger(__name__)


def judge_response(
    task_prompt: str,
    response_text: str,
    context: str = "",
    *,
    settings: Settings | None = None,
) -> list[CriterionScore]:
    """Score one response on the rubric criteria with the Haiku judge.

    Args:
        task_prompt: The original task.
        response_text: The agent response to score.
        context: Retrieved KB context the answer should rely on.
        settings: Injectable settings (defaults to the cached singleton).

    Returns:
        One :class:`CriterionScore` per rubric criterion, or an empty list if
        the judge call or its output could not be used (graceful degradation).
    """
    settings = settings or get_settings()
    prompt = build_judge_prompt(task=task_prompt, response=response_text, context=context)

    for attempt in range(max(1, settings.max_retries)):
        try:
            completion = generate(
                prompt,
                system=JUDGE_SYSTEM,
                provider="haiku",
                cache_system=True,
                settings=settings,
            )
        except LLMError as exc:
            log.warning("judge: provider call failed: %s; degrading to no judge", exc)
            return []

        scores = _parse_scores(completion.text)
        if scores:
            return scores
        log.warning(
            "judge: could not parse JSON (attempt %d/%d)", attempt + 1, settings.max_retries
        )

    log.warning("judge: giving up after %d attempts; degrading to no judge", settings.max_retries)
    return []


def _parse_scores(text: str) -> list[CriterionScore]:
    """Parse the judge's JSON into clamped ``CriterionScore`` objects.

    Tolerates prose/code fences by extracting the outermost ``{...}``. Returns an
    empty list on any structural problem so the caller can retry or degrade.
    """
    blob = _extract_json_object(text)
    if blob is None:
        return []
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []

    scores: list[CriterionScore] = []
    for name in CRITERION_NAMES:
        entry = data.get(name)
        if not isinstance(entry, dict) or "score" not in entry:
            return []  # missing a required criterion -> treat as unparseable
        try:
            raw = float(entry["score"])
        except (TypeError, ValueError):
            return []
        scores.append(
            CriterionScore(
                name=name,
                score=_clamp(raw),
                justification=entry.get("justification"),
            )
        )
    return scores


def _extract_json_object(text: str) -> str | None:
    """Return the substring from the first ``{`` to the last ``}``, or None."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


def _clamp(value: float) -> float:
    """Clamp a score into [0.0, 1.0] (judges occasionally drift out of range)."""
    return max(0.0, min(1.0, value))
