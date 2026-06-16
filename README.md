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

See [`CLAUDE.md`](CLAUDE.md) for the full specification and
[`docs/architecture.md`](docs/architecture.md) for the architecture write-up.

![Architecture](docs/diagram.png)

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
python main.py run "Your task"  # full critique->revise loop on one task
pytest
```

Prerequisite: [Ollama](https://ollama.com) running locally with the model
pulled — `ollama pull qwen2.5:7b`.

## CLI

```bash
python main.py run "<task>"                    # v1 -> feedback -> v2 on one task
python main.py evals --dataset golden --loop   # score a frozen dataset (v1 vs v2)
python main.py validate-judge --limit 5        # judge agreement vs golden (uses Haiku)
python main.py report --gate                   # metrics + charts + regression gate
python main.py demo                            # build the committed read-only demo
```

The agent runs on local Ollama (free); the Haiku judge is sampled and opt-in, so
the default path costs nothing (`--no-judge` / `judge_sample_rate=0`).

## App & UI (local)

The Streamlit UI talks to the agent **only** through the FastAPI boundary, so run
both processes (two terminals):

```bash
# Terminal 1 — API
uvicorn app.main:app --port 8000

# Terminal 2 — UI
streamlit run ui/streamlit_app.py
```

Submit a task in the UI to watch the pipeline execute **stage by stage in real
time** — retrieve → generate (v1) → deterministic evaluation → rule-based feedback
→ revise (v2) → re-evaluation — each stage showing its model, latency, FREE/PAID
cost chip, and intermediate output as it completes. A **model toggle** runs the
same task on the free local model (`qwen2.5:7b`) or on Claude Haiku. The **Results
explorer** tab browses stored experiments and renders the four charts.

Endpoints: `GET /health`, `POST /run` (`{"prompt": "...", "do_judge": false}`),
`POST /run/stream` (same body; streams one Server-Sent Event per pipeline stage
for the live view), `GET /results`.

> The deployed/public demo is **read-only** (`EVAL_LOOP_DEMO_MODE=1`, see below):
> it serves precomputed results and makes **zero live model calls**, so it is free
> and never exposes an API key. The real-time live mode is intended to run locally.

![Live mode demo](docs/live-mode.gif)

<!-- Record a short screen capture of one live run and save it as docs/live-mode.gif;
     the link above renders once the file exists. -->

## Deployed demo (zero cost)

The public demo runs on Streamlit Community Cloud with **no API calls and no
Ollama** — it serves a committed, read-only snapshot:

1. Build the snapshot locally (free, deterministic-only): `python main.py demo`.
   This writes `data/demo_results.db` and `reports/output/demo/` (both committed).
2. Deploy `ui/streamlit_app.py` on Streamlit Community Cloud.
3. Set the secret / env var **`EVAL_LOOP_DEMO_MODE=1`**. Demo mode hides the run
   form and reads only the committed snapshot + precomputed charts.

Locally you can preview demo mode the same way (Ollama and API key not required):

```bash
$env:EVAL_LOOP_DEMO_MODE = "1"   # PowerShell
streamlit run ui/streamlit_app.py
```

Optional env vars: `EVAL_LOOP_API_URL` (default `http://localhost:8000`) points the
live UI at a non-default API host/port.

## Results

All numbers below are **offline evaluation on the committed golden set** and are
reproducible from this repo (`python main.py demo`). The eval set is **fixed**;
only the agent changes — that is what makes the improvement measurable. N is
small, so the trend is **directional, not production-grade significant** — stated
honestly per `CLAUDE.md` §7.

**Held-out `test` split — critique→revise loop (N = 6, deterministic checks, free):**

| Metric | v1 | v2 | Δ |
|---|---|---|---|
| Mean overall score | 0.843 | 0.880 | **+0.037** |
| Pass rate (≥ 0.7) | 0.833 | 1.000 | **+0.167** |

Per-criterion means show *where* the gain comes from — the loop fixes coverage
without sacrificing grounding:

| Criterion | v1 | v2 | Δ |
|---|---|---|---|
| coverage | 0.556 | 0.722 | **+0.166** |
| grounding | 0.815 | 0.795 | −0.020 |
| length | 1.000 | 1.000 | 0.000 |
| non_empty | 1.000 | 1.000 | 0.000 |

The dev split (N = 17) shows the same direction (delta **+0.042**), with **zero
regressions** on either split: most v1 answers already pass on this KB, so the
loop engages only the weak ones and the framework *measures* that honestly rather
than inflating it (no best-of gate; the judge never runs inside the loop).

**Failure modes on v1** (what the agent gets wrong before revision): `low_grounding`
× 1, `missing_coverage` × 1 (1 of 6 v1 answers failed) — exactly the coverage gap
the loop then closes.

**Judge validation** (live Haiku, N = 5, cached rubric): ranking accuracy **1.00**,
good-vs-bad pass rate **1.00**, mean good **0.93** vs mean bad **0.30** — the judge
reliably ranks reference answers above degraded (off-topic / truncated) variants,
so its scores are trustworthy before being relied on (directional at N = 5).

Charts (regenerated by `python main.py report`) live in
[`reports/output/demo/`](reports/output/demo/): score trend, v1-vs-v2,
per-criterion breakdown, and failure-mode counts.

**Test suite:** 96 tests pass, `ruff` clean.

## Cost

Realized spend for the entire build is **far under the $3 target — effectively a
few cents at most** (CLAUDE.md §2 hard ceiling $10):

| Activity | Provider | Cost |
|---|---|---|
| The agent (generate + revise, all runs) | Ollama `qwen2.5:7b`, local | **$0** |
| Deterministic checks + metrics + charts | pure Python / pandas | **$0** |
| Demo build (`python main.py demo`) | local, deterministic-only | **$0** |
| Test suite (96 tests) | offline (model calls stubbed) | **$0** |
| Deployed Streamlit demo | committed snapshot, no live calls | **$0 ongoing** |
| Judge validation (`validate-judge --limit 5`) | Haiku 4.5, ~15 small calls, cached rubric | **≈ a fraction of a cent** |

The only paid calls in the whole project are the sampled Haiku judge, which is
opt-in and runs on a small subset with a cached system/rubric prompt (~90%
cached-input savings). The default path — agent on local Ollama, deterministic
checks — costs nothing, and the deployed demo makes **zero** live API calls.

## Status

Day 7 (refactor, docs, diagram, results, cost check) in progress. See `CLAUDE.md`
§8 for the day-by-day plan and §11 for the progress log.
