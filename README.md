# eval-loop-agent

A self-improving agent with an evaluation-driven feedback loop.

Given a task, the agent generates a response (grounded in a small local
knowledge base), evaluates it against a rubric (deterministic checks + a
sampled LLM judge), generates structured feedback, revises the response, and
compares versions — while an evaluation framework measures quality across a
fixed dataset and tracks **measurable improvement** over iterations.

> "Self-improving" here means the agent improves its *output* within a task via
> a critique-and-revise loop, and across experiments we track and tune its
> *prompts*. **Model weights are frozen** — this is program/response
> optimization, not retraining. It is offline evaluation on synthetic + golden
> traffic, not live production monitoring.

See [`CLAUDE.md`](CLAUDE.md) for the full specification.

## Stack

Python 3.13 · LangGraph · Ollama `qwen2.5:7b` (free local agent) · Anthropic
Claude Haiku (sampled judge) · Pydantic v2 · FastAPI · SQLite · Streamlit ·
pandas · matplotlib · httpx.

## Quickstart

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
copy .env.example .env          # then fill in ANTHROPIC_API_KEY (judge only)

python main.py --hello          # one local Ollama completion (free)
python main.py run "Your task"  # task -> grounded response -> trace -> SQLite
pytest
```

Prerequisite: [Ollama](https://ollama.com) running locally with the model
pulled — `ollama pull qwen2.5:7b`.

## Status

Day 1 (foundation + vertical slice) in progress. See `CLAUDE.md` §8 for the
day-by-day plan and §11 for the progress log.
