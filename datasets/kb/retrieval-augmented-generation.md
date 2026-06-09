# Retrieval-Augmented Generation (RAG)

Retrieval-augmented generation grounds a language model's answer in documents
fetched at query time, instead of relying only on what the model memorized
during training.

## What RAG is

A RAG system has two stages. First, a retriever searches a knowledge base for
passages relevant to the user's question. Second, the generator (the language
model) is given those passages as context and asked to answer using them. The
model's job shifts from "recall a fact" to "read the provided text and respond",
which is a task it does far more reliably.

## Why grounding matters

Grounding means the answer is supported by the retrieved context rather than
invented. A grounded answer can be traced back to a source passage, which makes
it checkable and reduces hallucination. When the retrieved context does not
contain the answer, a well-behaved RAG system says it does not know rather than
guessing, because a confident wrong answer is worse than an honest gap.

## When to use RAG

Use RAG when answers depend on a specific, changing, or private corpus that the
base model was not trained on: product docs, internal policies, or a curated
knowledge base. Avoid it when the task is pure reasoning or general knowledge
the model already handles well, since retrieval then adds latency and noise
without improving the answer.

## Retrieval methods

Retrieval ranges from simple keyword overlap (cheap, transparent, no training)
to dense embedding similarity (handles synonyms, needs a vector store). For a
small knowledge base, keyword or token-overlap scoring is often enough and keeps
the system free and fully deterministic, which makes evaluation reproducible.
