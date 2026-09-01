# Contributing

Thanks for your interest in `eval-loop`. This is a small, focused project;
contributions that keep it small and focused are the most welcome.

## Setup

Requires Python 3.13 and [Ollama](https://ollama.com) for the agent runtime.

```bash
ollama pull qwen2.5:7b

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev]"
```

No API key is needed for development. `ANTHROPIC_API_KEY` is only required to
run the LLM judge (`python main.py validate-judge`), which is the one paid path
in the project.

## Before you open a pull request

```bash
ruff check .
pytest
```

Both must be clean. **The test suite runs fully offline** — no Ollama, no API
key, no network. If a change makes a test require a live model call, that is a
bug in the test: stub the model boundary instead. CI runs with no API key
present, so a test that needs one will fail there.

## Merge rule

Every change lands through a pull request on `main`. A PR needs:

- **one approving review**, and
- **a green CI run** (the `test` check: `ruff` then `pytest`).

Direct pushes to `main` are blocked.

## Coding standards

The full standards live in [`CLAUDE.md`](CLAUDE.md) §6. In short:

- Full type hints on every function signature and return.
- Pydantic v2 models for anything crossing a boundary — no raw dicts passed around.
- One responsibility per module. A file's purpose should be obvious on opening it.
- Docstrings on every public function and class.
- Comments explain *why*, not *what*.
- Every model call goes through `agents/llm.py`; every database access goes
  through `storage/db.py`. No exceptions — these are the seams that keep the
  project testable and swappable.
- No secrets in code. Read configuration from `.env` via `config/settings.py`.

## Respect the cost rules

This project is deliberately cheap to run (see [`CLAUDE.md`](CLAUDE.md) §2).
Please keep it that way:

- The agent's default runtime is the local, free model. Don't route high-volume
  work through a paid API.
- Prefer deterministic checks wherever there is a structural or ground-truth
  answer. Reserve the LLM judge for genuinely subjective quality.
- The judge never runs inside the improvement loop, and it stays sampled.

## Scope

The stack is intentionally locked (see [`CLAUDE.md`](CLAUDE.md) §3). Pull
requests that add a database engine, a second web framework, container
tooling, or a cloud dependency will likely be declined — not because the idea is
bad, but because keeping the surface small is a goal of the project. If you want
to propose one, please open an issue first so we can discuss it before you spend
time on code.

## Reporting a problem

Open an issue with what you ran, what you expected, and what happened. If it
involves the agent's output, include the retrieved doc IDs and the scores — both
are printed by `python main.py run`.
