# LLM-as-a-Judge

LLM-as-a-judge uses a language model to score another model's output against a
rubric. It is the practical way to measure subjective quality — correctness,
completeness, clarity — that deterministic checks cannot capture.

## What it is

A judge model receives the task, the response, and a rubric describing the
scoring criteria, then returns a score per criterion with a short
justification. Because the rubric is fixed, the judge applies the same standard
to every response, which makes scores comparable across runs.

## Known biases

LLM judges have measurable biases. Position bias favors whichever answer appears
first when comparing two. Length bias rewards longer answers regardless of
quality. Self-preference bias favors text written in the judge's own style.
Verbosity and politeness can inflate scores. Knowing these biases is what lets
you design around them, for example by randomizing order and capping length.

## Validating the judge

A judge's scores are only trustworthy after you validate them against a
human-verified golden set. Compute agreement between the judge and the human
labels, then refine the rubric until agreement is high. Only then do the judge's
scores on unlabeled data mean anything. Skipping this step produces numbers that
look rigorous but measure nothing.

## Rubric design

A good rubric names each criterion, defines what a high and low score look like,
and asks for a brief justification before the score. Concrete anchors ("cites
the source" vs "makes unsupported claims") reduce ambiguity and make the judge
more consistent. Keep criteria few and independent so scores are interpretable.

## Controlling cost

Judging with a paid model is the expensive part of evaluation, so sample it:
run deterministic checks on the full dataset for free and send only a fixed
fraction to the LLM judge. Caching the rubric and system prompt across the batch
cuts most of the remaining input cost.
