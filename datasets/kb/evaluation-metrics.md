# Evaluation Metrics

Evaluation turns a non-deterministic agent's behavior into numbers you can
trust and track. Without measurement there is no basis to claim a change made
the agent better.

## Deterministic vs criteria-based evaluation

Deterministic checks have a structural or ground-truth answer: did the response
cite the retrieved context, is it valid format, does it cover the required
points, is it within length bounds. They are free, fast, and reproducible.
Criteria-based evaluation handles subjective quality that has no single correct
answer and needs a judge. The rule of thumb is cheapest-first: use deterministic
checks wherever a structural answer exists, and reserve the LLM judge for
genuinely subjective criteria.

## Per-criterion and overall scores

Rather than one opaque number, score each criterion separately (for example
correctness, completeness, clarity) on a fixed scale, then combine them into an
overall score. Per-criterion scores show *where* a response is weak, which is
what feedback and improvement act on.

## Pass rate and threshold

A pass rate is the fraction of responses whose overall score clears a fixed
threshold. It summarizes a dataset in one interpretable number and makes
regressions visible: if a change drops the pass rate, it made things worse.

## Improvement delta

The improvement delta is the average difference between a revised response and
its original on the same fixed dataset. A positive mean delta is the core
evidence that a feedback loop works. It is meaningful only when the evaluation
set is held fixed and only the agent changes.

## Sample-size honesty

At small dataset sizes, report N alongside any metric and treat trends as
directional, not production-grade significant. Claiming statistical
significance from twenty examples overstates the evidence; saying "v2 beats v1
on N=20, directionally" is honest and still useful.
