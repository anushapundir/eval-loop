# Agent Design

An agent is a language model wrapped in structure: a control flow that decides
what step to take next, tools it can call, and state that carries information
between steps. Good agent design keeps these concerns separate so each can be
tested and improved on its own.

## The agent loop

At its core an agent runs a loop: read the current state, decide on an action
(answer, call a tool, or revise), execute it, update the state, and repeat until
a stop condition. Making the loop explicit — rather than burying it in one giant
prompt — is what makes the agent's behavior observable and debuggable.

## Tool use

Tools let the agent reach beyond the model's parameters: a retrieval tool reads
a knowledge base, a calculator does arithmetic, an API call fetches live data.
A tool has a clear input and output contract, and the agent decides when to call
it. Retrieval is the most common tool, because grounding answers in fetched text
is more reliable than recalling them.

## State

State is the typed object threaded through every step: the task, retrieved
context, intermediate responses, scores, and a counter that bounds any loop.
Keeping state in one validated object — instead of scattered variables — makes
each step a pure function of the state in and the state out, which is easy to
trace and reason about.

## Orchestration graphs

A graph framework models the agent as nodes (steps) and edges (transitions),
with a shared state object. This makes branching and loops first-class: a
conditional edge can route back to revise or forward to finish. The graph is the
skeleton; the prompts and tools are the muscle.

## Observability

Every meaningful step should emit a trace: which node ran, what it received,
what it produced, and how long it took. Traces let you reconstruct and audit a
run offline, which is essential when the underlying model is non-deterministic.
