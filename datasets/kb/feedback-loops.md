# Feedback Loops and Self-Improvement

A feedback loop lets an agent improve its own output: it generates a response,
evaluates it, turns the evaluation into actionable feedback, and revises. The
model weights never change — this is program and response optimization, not
retraining.

## The critique-and-revise loop

The loop has four moves: generate an initial response, evaluate it against a
rubric, produce structured feedback naming concrete weaknesses, and revise the
response using that feedback. The revised version is then re-scored so the
improvement can be measured rather than assumed.

## Self-refinement (Reflexion)

This pattern is sometimes called self-refinement or Reflexion: a model reflects
on its own output, articulates what is wrong, and tries again. The key is that
the critique is specific and actionable ("missing the cost tradeoff", not "make
it better"), because vague feedback produces vague revisions.

## Stop criteria

A loop that can revise forever will waste cost and can even degrade quality.
Bound it with a maximum number of iterations, and optionally stop early once the
score crosses a threshold or stops improving between rounds. A hard
max-iterations cap is the simplest safe stop.

## Avoiding overfitting

If the same examples drive both the feedback and the measurement, the agent can
learn to please the metric rather than genuinely improve. Hold out a split that
the feedback step never sees, and report results on that held-out set, so the
measured improvement reflects real gains rather than memorized fixes.

## Human in the loop

An optional human gate can approve or reject a revision before it is accepted.
This keeps a person in control of consequential changes while still letting the
automated loop do the bulk of the diagnostic and drafting work.
