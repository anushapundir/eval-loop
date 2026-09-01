# eval-loop-agent

An AI agent that improves its own answers through a closed evaluation loop.

Give it a task. It retrieves from a local knowledge base, writes an answer,
scores that answer against a rubric, turns the score into structured feedback,
rewrites the answer, and scores it again — then reports whether it actually got
better across a fixed dataset.

> **What "self-improving" means here.** The agent improves its *output* within a
> task, and across experiments we track and tune its *prompts*. **Model weights
> are frozen** — this is program optimization, not retraining. It is offline
> evaluation on synthetic and golden data, not live production monitoring.

The agent runs on a **local, free** model by default. The only paid calls are a
sampled Claude Haiku judge, which is opt-in.

---

## Quickstart

**Prerequisites:** Python 3.13 and [Ollama](https://ollama.com) running locally.

```bash
ollama pull qwen2.5:7b
```

**Install:**

```bash
git clone https://github.com/anushapundir/eval-loop-agent.git
cd eval-loop-agent

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

pip install -e ".[dev]"
```

**Run your first loop:**

```bash
python main.py run "What is retrieval-augmented generation?"
```

That's it. No API key needed — the default path is entirely local and free.

An API key is only required for the LLM judge. To enable it, copy
`.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

---

## What you'll see

`python main.py run` walks the full loop and prints each stage:

```
retrieve   →  which knowledge-base docs were pulled in
generate   →  v1, the first answer
evaluate   →  v1 scored on grounding, coverage, length, non-empty
feedback   →  what specifically was wrong, e.g. which key points are missing
revise     →  v2, the rewritten answer
evaluate   →  v2 scored the same way, plus the delta
```

If v1 already passes, the loop short-circuits and carries it forward rather than
revising for the sake of it. That is deliberate: the framework measures
improvement honestly instead of manufacturing it.

---

## CLI

| Command | What it does |
|---|---|
| `python main.py run "<task>"` | Full v1 → feedback → v2 loop on one task |
| `python main.py evals --dataset golden --loop` | Score a frozen dataset, v1 vs v2 |
| `python main.py report --gate` | Metrics, charts, and a regression gate |
| `python main.py validate-judge --limit 5` | Judge agreement vs golden (**uses Haiku — paid**) |
| `python main.py demo` | Rebuild the committed read-only demo snapshot |
| `python main.py --hello` | One local Ollama completion, to check your setup |

Everything except `validate-judge` is free by default. Add `--no-judge` (or set
`judge_sample_rate=0`) to guarantee no paid calls.

---

## Run the web app

The Streamlit UI talks to the agent **only** through the FastAPI boundary, so
run both processes in two terminals:

```bash
# Terminal 1 — API
uvicorn app.main:app --port 8000

# Terminal 2 — UI
streamlit run ui/streamlit_app.py
```

Submit a task and watch the pipeline execute **stage by stage in real time** —
each stage showing its model, latency, a FREE/PAID cost chip, and its
intermediate output as it completes. A model toggle runs the same task on the
free local model or on Claude Haiku. The **Results explorer** tab browses stored
experiments and renders the charts.

**Endpoints:** `GET /health` · `POST /run` · `POST /run/stream` (one
Server-Sent Event per stage) · `GET /results`

Optional: `EVAL_LOOP_API_URL` (default `http://localhost:8000`) points the UI at
a different API host.

---

## How it works

```
User → UI (Streamlit) → FastAPI → Agent (generate v1)
     → Evaluation (score v1) → Feedback (structured critique) → Agent (revise → v2)
     → Evaluation (score v2) → Comparison → Metrics → SQLite → back to the UI
```

Layers are separated so each can be understood and changed on its own; **SQLite
is the integration boundary**, which is what lets evaluation run offline against
stored data.

- `agents/` — the system under test (LangGraph graph + retrieval tool)
- `evaluators/` — the measuring instrument (deterministic checks + LLM judge)
- `feedback/` — diagnoses failures and drives the revision
- `storage/` — the only module that touches SQLite
- `reports/` — pandas metrics and matplotlib charts
- `app/` · `ui/` — thin API and presentation layers

Evaluation is **hybrid and cheapest-first**: free deterministic checks
(grounding, coverage, format, length) do the work wherever there is a structural
or ground-truth answer, and a sampled LLM judge handles only subjective quality.
The judge is validated against a human-verified golden set before its scores are
trusted at all.

![Architecture](docs/diagram.png)

Full write-up: [`docs/architecture.md`](docs/architecture.md). The complete
project specification and build log is [`CLAUDE.md`](CLAUDE.md).

---

## Results

All numbers below are **offline evaluation on the committed golden set** and are
reproducible from this repo (`python main.py demo`). The eval set is **fixed**;
only the agent changes — that is what makes the improvement measurable. N is
small, so the trend is **directional, not production-grade significant**.

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

**Failure modes on v1** (what the agent gets wrong before revision):
`low_grounding` × 1, `missing_coverage` × 1 (1 of 6 v1 answers failed) — exactly
the coverage gap the loop then closes.

**Judge validation** (live Haiku, N = 5, cached rubric): ranking accuracy
**1.00**, good-vs-bad pass rate **1.00**, mean good **0.93** vs mean bad
**0.30** — the judge reliably ranks reference answers above degraded (off-topic
or truncated) variants, so its scores are trustworthy before being relied on
(directional at N = 5).

Charts (regenerated by `python main.py report`) live in
[`reports/output/demo/`](reports/output/demo/): score trend, v1-vs-v2,
per-criterion breakdown, and failure-mode counts.

**Test suite:** 101 tests pass, `ruff` clean.

---

## Cost

The entire build cost **well under a dollar** — effectively a few cents.

| Activity | Provider | Cost |
|---|---|---|
| The agent (generate + revise, all runs) | Ollama `qwen2.5:7b`, local | **$0** |
| Deterministic checks + metrics + charts | pure Python / pandas | **$0** |
| Demo build (`python main.py demo`) | local, deterministic-only | **$0** |
| Test suite (101 tests) | offline (model calls stubbed) | **$0** |
| Deployed Streamlit demo | committed snapshot, no live calls | **$0 ongoing** |
| Judge validation (`validate-judge --limit 5`) | Haiku, ~15 small calls, cached rubric | **≈ a fraction of a cent** |

The only paid calls are the sampled Haiku judge — opt-in, run on a small subset,
with a cached system/rubric prompt (~90% cached-input savings). The default path
costs nothing, and the deployed demo makes **zero** live API calls.

---

## Deployed demo (zero cost)

The public demo runs on Streamlit Community Cloud with **no API calls and no
Ollama** — it serves a committed, read-only snapshot:

1. Build the snapshot locally (free): `python main.py demo`. This writes
   `data/demo_results.db` and `reports/output/demo/`, both committed.
2. Deploy `ui/streamlit_app.py` on Streamlit Community Cloud.
3. Set the env var **`EVAL_LOOP_DEMO_MODE=1`**. Demo mode hides the run form and
   reads only the committed snapshot and precomputed charts.

Preview demo mode locally (no Ollama or API key needed):

```bash
$env:EVAL_LOOP_DEMO_MODE = "1"   # PowerShell
streamlit run ui/streamlit_app.py
```

The real-time live mode is intended to run locally, where the model calls are
free and no key is exposed.

---

## Contributing

Contributions are welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md). Every
change goes through a pull request that needs one approving review and a green
CI run.

## License

[MIT](LICENSE)
