"""Agent prompts — versioned, so improvement and experiments can swap them.

Prompts live here (not inline in the graph) for two reasons: the self-improvement
work edits prompts, and Day 5 experiments compare prompt versions. Each prompt
carries an explicit version string that flows into ``Experiment.prompt_version``.

The generation prompt instructs the model to answer *from the retrieved context*
and to admit when the context is insufficient — this is what makes the Day 3
grounding check meaningful and curbs hallucination.
"""

from __future__ import annotations

from evaluators.rubric import CRITERION_NAMES, RUBRIC_VERSION, render_rubric

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


# --- Revision prompt (drives the self-improvement loop) --------------------
# The revise step rewrites v1 into v2 using the reviewer feedback, staying
# grounded in the same retrieved context. It deliberately takes only the
# previous answer + feedback (never the golden reference answer) so the loop's
# improvement is real, not leaked from labels (CLAUDE.md §7).
REVISION_PROMPT_VERSION = "v1"

REVISION_SYSTEM = (
    "You are revising a previous answer about AI evaluation and agent "
    "engineering. Apply the reviewer's feedback to improve the answer, using "
    "ONLY the provided context. Keep what was already correct, fix what the "
    "feedback identifies, and do not invent facts or use information outside the "
    "context. If the context does not contain the answer, say you don't have "
    "enough information rather than guessing."
)

REVISION_TEMPLATE = """Use ONLY the following context to revise the answer.

Context:
{context}

Question: {question}

Previous answer:
{previous_answer}

Reviewer feedback to address:
{feedback}

Write an improved answer that addresses the feedback and stays grounded in the
context:"""


def build_revision_prompt(
    *, question: str, context: str, previous_answer: str, feedback: str
) -> str:
    """Render the revision prompt that turns v1 into v2.

    Args:
        question: The user's task prompt.
        context: Retrieved KB text the answer must stay grounded in.
        previous_answer: The v1 response being revised.
        feedback: The structured reviewer feedback (rendered text) to apply.

    Returns:
        The full user prompt to send to the model. The golden ``expected``
        answer is intentionally not a parameter, so it can never leak in.
    """
    return REVISION_TEMPLATE.format(
        context=context if context.strip() else "(no relevant context found)",
        question=question,
        previous_answer=previous_answer,
        feedback=feedback if feedback.strip() else "(no specific feedback)",
    )


# --- Judge prompt (LLM-as-judge, Haiku) ------------------------------------
# The version pins both the prompt template and the rubric it embeds, so a score
# change can be attributed to either. The system block is static (rubric + JSON
# contract) so it caches across the judging batch (cache_system=True).
JUDGE_PROMPT_VERSION = f"judge-v1+rubric-{RUBRIC_VERSION}"

JUDGE_SYSTEM = (
    "You are a strict, fair evaluator of answers about AI evaluation and agent "
    "engineering. Score the answer on each criterion from 0.0 (worst) to 1.0 "
    "(best), judging only the answer's quality for the task — not its length or "
    "politeness.\n\n"
    "Criteria:\n"
    f"{render_rubric()}\n\n"
    "Respond with ONLY a JSON object, no prose and no code fences, mapping each "
    "criterion name to an object with a numeric \"score\" in [0.0, 1.0] and a "
    "short \"justification\". Use exactly these keys: "
    f"{', '.join(CRITERION_NAMES)}.\n"
    'Example: {"correctness": {"score": 0.8, "justification": "..."}, '
    '"completeness": {"score": 0.6, "justification": "..."}, '
    '"clarity": {"score": 0.9, "justification": "..."}}'
)

JUDGE_TEMPLATE = """Task:
{task}

Retrieved context the answer should rely on:
{context}

Answer to score:
{response}

Return the JSON object now."""


def build_judge_prompt(*, task: str, response: str, context: str = "") -> str:
    """Render the judge's user prompt for one task/response pair.

    Args:
        task: The original task prompt.
        response: The agent response being scored.
        context: The retrieved KB context the answer should rely on.

    Returns:
        The user prompt to send to the judge (the rubric lives in the system
        block so it can be cached across the batch).
    """
    return JUDGE_TEMPLATE.format(
        task=task,
        context=context if context.strip() else "(no context was retrieved)",
        response=response,
    )
