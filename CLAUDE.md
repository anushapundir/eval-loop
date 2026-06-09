# CLAUDE.md — eval-loop-agent

**Project:** `eval-loop-agent` — a Self-Improving Agent with an Evaluation-Driven Feedback Loop.

This file is the **single source of truth** for Claude Code. Build the project from this document. When anything conflicts, this file wins. Do not expand scope, swap the stack, or skip the cost rules without explicit human approval. Workflow is **Path A: propose → human understands → human approves → execute.**

---

## 1. What this project is

`eval-loop-agent` is an AI agent that improves its own output through a closed evaluation loop. Given a task, it generates a response, evaluates that response against a rubric, generates structured feedback, revises the response, and compares the versions — while an evaluation framework measures quality across a fixed dataset and tracks **measurable improvement** over iterations.

The agent lifecycle:

1. Receive a user task.
2. Generate an initial response (grounded in a small local knowledge base via a retrieval tool).
3. Evaluate its own response (deterministic checks + a sampled LLM judge).
4. Generate structured feedback from the evaluation.
5. Revise the response using the feedback.
6. Compare versions (v1 vs v2) and re-score.
7. Track evaluation metrics across the dataset and across experiment iterations.
8. Demonstrate measurable improvement (average v2 score > v1 score), shown in a dashboard.

**What "self-improving" means here (state this precisely):** the agent improves its *output* within a task via a critique-and-revise loop, and across experiments we track and can tune the agent's *prompts*. The **model weights are frozen** — this is program/response optimization, not retraining. Honest framing: it is **offline evaluation on synthetic + golden traffic**, not live production monitoring.

**Showcases:** Agent Design · Evaluation Systems · Feedback Loops · AI Engineering · System Design · Experimentation · Observability · Production Engineering Practices.

---

## 2. COST RULES (highest priority — every decision respects these)

- **Target total cost: under $3. Hard ceiling: $10.** No ongoing cost after the build.
- **Default agent runtime is Ollama (`qwen2.5:7b`), local, free.** Do NOT route the high-volume agent through the paid API by default.
- **Claude Haiku is used sparingly** — only as the LLM judge, and only on a **sampled subset**. Cache the rubric/system prompt (≈90% cached-input savings).
- **Deterministic checks are free** — use them for anything with a ground-truth or structural answer. LLM judging only for subjective quality.
- **Small datasets**: golden ~20–40 (human-verified), synthetic ~100–300 inputs (generated once, frozen). The agent + deterministic checks are free, so run the deterministic metric on the *full* set; sample only the paid judge.
- **A config switch (`MODEL_PROVIDER`)** allows an "all-Haiku" mode for higher quality when desired; default is cost-safe (agent=Ollama, judge=Haiku-sampled).
- **Deployed demo makes zero live API calls** (precomputed results only) — see §6 Day 6.

Cost estimate at default settings: ≈ **$0.50–1.50 total**.

---

## 3. Technology stack (locked — keep minimal, no extras)

| Tool | Role |
|---|---|
| Python 3.13 | language/runtime (Windows 11) |
| LangGraph | agent orchestration (graph: nodes/edges/state); the critique→revise loop |
| Ollama `qwen2.5:7b` | **default** agent runtime, local, free |
| Anthropic Claude Haiku 4.5 | LLM judge (sampled); optional generation in all-Haiku mode |
| Pydantic v2 | data validation / all data models |
| FastAPI | thin local application/API layer exposing the agent |
| SQLite | storage (traces, evals, experiments) — not Postgres |
| Streamlit | UI + deployed read-only demo |
| Pandas | metrics aggregation |
| Matplotlib | report/dashboard charts |
| httpx | HTTP client where needed |

**Do NOT add:** Postgres, Docker, additional web frameworks, LangSmith (build tracing yourself; Phoenix optional for validation only), cloud (AWS is a possible Phase 2, not now), Reddit/extra data sources.

---

## 4. Architecture

### 4.1 High-level architecture (layers)

```
┌──────────────────────────────────────────────────────────────┐
│  USER LAYER            a person submitting a task              │
├──────────────────────────────────────────────────────────────┤
│  UI LAYER (ui/)        Streamlit: submit task, view v1→v2,     │
│                        metrics, charts; read-only demo mode    │
├──────────────────────────────────────────────────────────────┤
│  APPLICATION LAYER     FastAPI (app/): thin endpoints that     │
│  (app/)                run the agent loop and return results   │
├──────────────────────────────────────────────────────────────┤
│  AGENT LAYER           LangGraph graph: generate → (loop)      │
│  (agents/)             critique → revise; retrieval tool;      │
│                        emits traces                            │
├──────────────────────────────────────────────────────────────┤
│  EVALUATION LAYER      deterministic checks + LLM judge +      │
│  (evaluators/)         rubric + runner → structured scores     │
├──────────────────────────────────────────────────────────────┤
│  FEEDBACK LAYER        turns evaluation into structured        │
│  (feedback/)           feedback and drives the revision        │
├──────────────────────────────────────────────────────────────┤
│  STORAGE LAYER         SQLite: traces, eval results,           │
│  (storage/)            experiments; the integration boundary   │
├──────────────────────────────────────────────────────────────┤
│  REPORTING LAYER       pandas + matplotlib: metrics,           │
│  (reports/)            comparisons, charts                     │
└──────────────────────────────────────────────────────────────┘
```

The **Storage Layer is the integration boundary**: every layer depends on the data *schema*, not on each other directly. This makes the system reproducible and lets evaluation run offline on stored data.

### 4.2 Request flow

```
User → UI (Streamlit) → FastAPI endpoint → Agent (generate initial response)
     → Evaluation (score v1) → Feedback (structured critique) → Agent (revise → v2)
     → Evaluation (score v2) → Comparison (v1 vs v2) → Metrics → Storage → Final output to UI
```

### 4.3 Data flow

- **Originates:** user task (UI/API) and dataset inputs (`datasets/`).
- **Validated:** at every boundary via Pydantic models (`storage/models.py`) — task in, response out, eval result, experiment record.
- **Stored:** SQLite (`storage/db.py`) — one row per trace, per eval result, per experiment.
- **Metrics generated:** `reports/metrics.py` aggregates stored eval results with pandas.
- **Reports generated:** `reports/charts.py` renders matplotlib charts; the UI reads stored results for display.
- **Fixed vs produced:** dataset inputs and golden labels are *fixed* (generated/authored once, frozen, committed); traces, scores, and experiment records are *produced fresh* each run.

---

## 5. File structure (every directory has one responsibility)

```
eval-loop-agent/
├── CLAUDE.md                 # this file — source of truth
├── README.md                 # setup, run, results, screenshots
├── pyproject.toml            # deps + project metadata
├── .env.example              # ANTHROPIC_API_KEY template (real .env gitignored)
├── .gitignore
├── main.py                   # CLI entry: run agent / run evals / run experiment
│
├── config/                   # configuration & cross-cutting setup
│   ├── settings.py           #   model provider switch, paths, thresholds, sample sizes
│   └── logging.py            #   centralized logging config
│
├── agents/                   # THE AGENT (system under test)
│   ├── llm.py                #   model abstraction: Ollama (default) <-> Haiku
│   ├── state.py              #   LangGraph state object (Pydantic)
│   ├── prompts.py            #   agent prompts (versioned; improvement edits these)
│   ├── tools.py              #   retrieval tool over the local knowledge base
│   └── graph.py              #   LangGraph graph: generate → critique → revise loop
│
├── evaluators/               # EVALUATION FRAMEWORK
│   ├── checks.py             #   deterministic checks (grounding, format, coverage, length)
│   ├── rubric.py             #   judging criteria (the rubric, as data)
│   ├── judge.py              #   LLM-as-judge (Haiku, sampled, cached)
│   ├── runner.py             #   runs all evaluators over traces → EvalResult
│   └── analysis.py           #   cluster failures into named modes
│
├── feedback/                 # FEEDBACK & IMPROVEMENT
│   ├── generator.py          #   structured feedback from an EvalResult
│   └── improve.py            #   revise a response using feedback (drives the loop)
│
├── datasets/                 # FIXED DATA (committed)
│   ├── kb/                   #   small local knowledge base (markdown docs)
│   ├── golden/               #   human-verified examples (task + correct answer/labels)
│   ├── synthetic/            #   generated-once eval inputs (frozen)
│   └── generate.py           #   dev-time generation spec + the prompt used (documented)
│
├── storage/                  # STORAGE LAYER
│   ├── models.py             #   Pydantic models: Task, AgentResponse, Trace, EvalResult, Experiment
│   └── db.py                 #   ONLY module that touches SQLite (create, read, write)
│
├── reports/                  # REPORTING LAYER
│   ├── metrics.py            #   pandas aggregation of stored results
│   ├── charts.py             #   matplotlib charts (trend, v1-vs-v2, per-criterion, failures)
│   └── output/               #   generated charts/reports (gitignored except demo set)
│
├── app/                      # APPLICATION LAYER
│   └── main.py               #   thin FastAPI app: run task / run experiment / fetch results
│
├── ui/                       # UI LAYER
│   └── streamlit_app.py      #   interactive demo + results explorer; read-only demo mode
│
├── tests/                    # unit tests for deterministic logic & schemas
│   ├── test_models.py
│   ├── test_checks.py
│   ├── test_db.py
│   └── test_graph.py
│
└── docs/                     # documentation
    ├── architecture.md       #   architecture write-up + diagram
    └── diagram.png           #   architecture diagram
```

**Why each top-level folder exists:**

- `config/` — one place for settings and logging so nothing is hardcoded; flipping `MODEL_PROVIDER` or a threshold touches one file.
- `agents/` — the system under test; isolated so it can be improved without disturbing evaluation.
- `evaluators/` — the measuring instrument; isolated so scoring is reproducible and trustable.
- `feedback/` — separates "diagnose + propose improvement" from both the agent (doer) and evaluator (judge); keeps the three roles distinct.
- `datasets/` — fixed inputs and ground truth, committed for reproducibility.
- `storage/` — the only place touching SQLite; the integration boundary; swappable in one file.
- `reports/` — analysis and visualization, reading only from storage.
- `app/` — thin API boundary so the UI and the agent are decoupled.
- `ui/` — presentation; talks to the API locally, reads stored results for the demo.
- `tests/` — guards the deterministic parts (schemas, checks, DB, graph wiring).
- `docs/` — architecture and diagrams for reviewers and interviews.

---

## 6. Coding standards

- **Full type hints** on every function signature and return.
- **Pydantic v2 models** for all data crossing a boundary (input, output, stored records). No raw dicts passed around.
- **Single responsibility**: small, focused functions; one job each. A file's purpose is obvious on opening it.
- **Naming**: descriptive, lowercase_with_underscores; modules named for their responsibility (`judge.py`, not `utils.py`).
- **Docstrings** on every public function/class: one line on what it does, plus args/returns when non-trivial.
- **Comments** explain *why*, not *what*.
- **Error handling + retries** on all model/API calls; never let one failed call crash a batch.
- **No secrets in code** — read from `.env` via `config/settings.py`.
- **Consistent formatting** (ruff/black-style). **Beginner-readable**: prefer clarity over cleverness.
- Every model call goes through `agents/llm.py`; every DB access through `storage/db.py`. No exceptions.

---

## 7. Evaluation framework (the heart)

**What is evaluated:** the quality of the agent's response to a task, on multiple criteria.

**Why:** the agent is non-deterministic; "correct" is graded, not binary. Without measurement there is no basis to claim improvement. Evaluation converts behavior into a number we can trust and track.

**How — a hybrid, cheapest-first:**
- **Deterministic checks** (`checks.py`, free): grounding (does the answer use the retrieved KB?), format validity, required-key-point coverage, length bounds. Used for anything with a structural/ground-truth answer.
- **LLM-as-judge** (`judge.py`, Haiku, sampled, cached): subjective quality — correctness, completeness, clarity — scored against the `rubric.py` criteria with a short justification.

**Metrics:** per-criterion scores (0–1 or 1–5), an overall score, pass-rate against a threshold, and the **improvement delta** (v2 − v1) averaged across the dataset.

**How improvement is measured:** the eval set is **fixed**; only the agent/response changes. Improvement = average v2 score > average v1 score on the dataset, and (across experiments) a rising trend. A **held-out split** the feedback step never sees prevents overfitting.

**Judge trust (non-negotiable):** validate the judge against the human-verified golden set — compute agreement; refine the rubric until agreement is high. Only then are judge scores meaningful. **Report N honestly**; at small N the trend is directional, not production-grade significant.

---

## 8. Day-by-day plan (7 days → complete, demo-ready)

Each day builds on the last. Build a **thin vertical slice first**, then thicken.

### Day 1 — Foundation, config, models, agent skeleton

- **Objective:** a clean, runnable skeleton with config, logging, validated data models, the model abstraction, and an empty agent graph.
- **Concepts:** project structure & separation of concerns, configuration management, Pydantic validation, LLM statelessness, model-provider abstraction.
- **Features:** folder tree; `config/settings.py` (paths, thresholds, sample sizes, `MODEL_PROVIDER`); `config/logging.py`; Pydantic models in `storage/models.py` (`Task`, `AgentResponse`, `Trace`, `EvalResult`, `Experiment`); `agents/llm.py` (Ollama default, Haiku optional) with retries; `agents/state.py`; `agents/graph.py` (skeleton with a single pass-through node); `main.py` CLI stub; `README.md`; `pyproject.toml`; `.env.example`.
- **Files created:** the full tree above (empty modules with docstrings) plus the files just listed implemented.
- **Architecture changes:** all layers stubbed; model abstraction and storage models established.
- **Deliverables:** `python main.py --hello` makes one successful local Ollama completion and logs it.
- **Validation:** config loads; `agents/llm.py` returns text from Ollama; `tests/test_models.py` passes; tree matches §5.
- **Success criteria:** skeleton runs end-to-end with a trivial LLM call; no hardcoded values.
- **Expected output:** a clean repo skeleton + one logged local completion.

### Day 2 — First working agent (initial response + tracing + storage)

- **Objective:** the agent takes a task and produces a grounded initial response, traced and stored.
- **Concepts:** the agent loop, LangGraph nodes/edges/state, prompt design, tool use (retrieval), tracing/observability.
- **Features:** `datasets/kb/` (3–6 markdown docs); `agents/tools.py` (simple keyword/embedding-free retrieval over the KB); `agents/prompts.py` (generation prompt); `agents/graph.py` (real `generate` node using retrieval); trace capture; `storage/db.py` (create tables, write trace).
- **Files created:** `agents/tools.py`, `agents/prompts.py`, updated `agents/graph.py`, `storage/db.py`, KB docs, `tests/test_graph.py`, `tests/test_db.py`.
- **Architecture changes:** Agent Layer + Storage Layer functional; tracing wired.
- **Deliverables:** run a task → grounded initial response → trace row in `data/traces.db`.
- **Validation:** run on 3 sample tasks; inspect rows with the **VS Code SQLite extension**; responses are non-empty and reference KB content.
- **Success criteria:** agent reliably produces and stores a traced response for any task.
- **Expected output:** working initial-response agent + stored traces.

### Day 3 — Evaluation framework + datasets (be thorough)

- **Objective:** build trustworthy evaluation: datasets, deterministic checks, LLM judge, scoring, storage, and judge validation.
- **Concepts:** eval vs testing, golden datasets, LLM-as-judge + biases + validation, deterministic vs criteria-based eval, rubric design, sample-size honesty.
- **How datasets are created:** `datasets/golden/` — Claude Code may *draft*, but the **human verifies every label**; ~20–40 examples, class/criteria-balanced, with edge cases. `datasets/synthetic/` — generated **once** via Claude Code into a file and **frozen**, ~100–300 varied inputs (vary tone/length/difficulty; include ambiguous cases). `datasets/generate.py` documents the generation prompt. Split into dev + held-out test.
- **How scoring works:** `evaluators/runner.py` reads traces, runs `checks.py` (deterministic) and `judge.py` (Haiku, **sampled**, rubric cached), writes `EvalResult` rows.
- **Why evaluation matters / where results stored:** see §7; results persist in SQLite for reproducible, offline re-scoring.
- **Features/files:** `evaluators/checks.py`, `rubric.py`, `judge.py`, `runner.py`, `analysis.py`; `datasets/` populated; `tests/test_checks.py`; a judge-vs-golden agreement script.
- **Architecture changes:** Evaluation Layer functional; results persisted.
- **Deliverables:** scores for the dataset in SQLite; a printed/stored judge-agreement number.
- **Validation:** deterministic checks pass on known cases; judge agreement with golden computed and reported; scores reproducible.
- **Success criteria:** every response gets a structured, trustworthy score; judge validated.
- **Expected output:** working evaluation framework + a validation metric.

### Day 4 — Feedback + self-improvement loop

- **Objective:** close the loop — generate structured feedback, revise the response, compare versions, demonstrate measurable improvement.
- **Concepts:** feedback loops, self-refinement (Reflexion), multi-step reasoning, closed-loop improvement, stop criteria, human-in-the-loop (optional gate).
- **Features:** `feedback/generator.py` (structured feedback from an `EvalResult`); `feedback/improve.py` (revise using feedback); `agents/graph.py` extended into `generate → evaluate → feedback → revise → re-evaluate` with a **max-iterations stop**; store v1, v2, and both scores; compute the delta.
- **Files created:** `feedback/generator.py`, `feedback/improve.py`, updated `graph.py`, storage updates.
- **Architecture changes:** Feedback Layer functional; the loop closes.
- **Deliverables:** one task shows v1 → feedback → v2, with average v2 > v1 on the dataset; loop terminates safely.
- **Validation:** on the dataset, mean v2 score > mean v1 score; loop respects max iterations; report N and the delta honestly.
- **Success criteria:** demonstrable, measurable improvement from the loop.
- **Expected output:** working self-improvement loop with before/after metrics.

### Day 5 — Experiment tracking, metrics, comparison, reporting

- **Objective:** track experiments, aggregate metrics, compare versions/iterations, generate reports.
- **Concepts:** experimentation, observability, metrics aggregation, reproducibility, fixed eval set, regression gating across iterations.
- **Features:** `Experiment` records in SQLite; `reports/metrics.py` (pandas aggregation: per-criterion, overall, delta, pass-rate); `reports/charts.py` (matplotlib: score trend across iterations, v1-vs-v2, per-criterion breakdown, failure-mode counts); a comparison routine; optional regression gate for prompt-level iterations.
- **Files created:** `reports/metrics.py`, `reports/charts.py`, storage updates for experiments.
- **Architecture changes:** Reporting Layer functional; experiments tracked.
- **Deliverables:** an experiment run → aggregated metrics + saved charts + a comparison report.
- **Validation:** charts render from stored data; metrics match a hand-check on a sample; comparison shows clear deltas.
- **Success criteria:** a reproducible experiment report with clear improvement metrics + visuals.
- **Expected output:** metrics + charts + comparison artifacts.

### Day 6 — FastAPI + Streamlit UI + demo readiness

- **Objective:** a thin API + an interactive UI, plus the zero-cost deployed demo.
- **Concepts:** application/API layer, UI layer, frontend/backend decoupling, precomputed deployment for zero ongoing cost.
- **Features:** `app/main.py` (FastAPI: `POST /run` runs the loop and returns v1/v2/metrics; `GET /results` reads stored experiments); `ui/streamlit_app.py` (interactive: enter a task, watch v1 → feedback → v2, view metrics/charts; plus a results explorer); wire Streamlit → FastAPI locally; build **read-only demo mode** that reads a committed `data/demo_results.db` for Streamlit Community Cloud.
- **Files created:** `app/main.py`, `ui/streamlit_app.py`, `data/demo_results.db` (committed), deployment notes in README.
- **Architecture changes:** Application + UI Layers functional; deploy path defined.
- **Deliverables:** local interactive app works; a **public read-only demo link** on the free tier, making no live API calls.
- **Validation:** locally, submit a task and see the full loop + metrics; the deployed link loads precomputed results with zero API calls.
- **Success criteria:** a clean, story-first demo (initial vs improved + metric trend) at zero ongoing cost.
- **Expected output:** working local app + a live public demo link.

### Day 7 — Refactor, docs, diagrams, tests, portfolio

- **Objective:** polish to production quality and prepare portfolio/interview materials.
- **Concepts:** production engineering practices, documentation, reproducibility, communicating results.
- **Features:** refactor for clarity; complete docstrings/type hints; finish tests; `docs/architecture.md` + diagram; final `README.md` (setup, run, results, screenshots, demo link); resume bullet + interview talking points (see §9); final cost check.
- **Files created:** `docs/architecture.md`, `docs/diagram.png`, finalized `README.md`, completed `tests/`.
- **Architecture changes:** none — hardening only.
- **Deliverables:** complete, documented, demo-ready repo with public link, results, and diagram.
- **Validation:** a fresh clone runs from the README; tests pass; demo link live; total spend under budget.
- **Success criteria:** a reviewer can understand the project in minutes, run it, and see measurable improvement; you can defend every part.
- **Expected output:** portfolio-ready project + resume line + interview talking points.

---

## 9. Interview readiness

**Demonstrates:** Agent Design (LangGraph multi-step graph + retrieval tool) · Evaluation Systems (hybrid deterministic + validated LLM-judge) · Feedback Loops (closed critique-revise-compare loop) · AI Engineering (reliability around a non-deterministic component) · System Design (clean layered architecture, storage as integration boundary) · Experimentation (fixed eval set, tracked iterations, held-out split) · Data Analysis (metrics + visual comparison).

**Talking points:**
- "Self-improving means the response and prompts improve from eval feedback — weights are frozen. Program optimization, not retraining."
- "I used deterministic checks wherever there was ground truth and an LLM judge only for subjective quality — cheaper and more reliable."
- "I validated the judge against a human-verified golden set before trusting any score."
- "The eval set is fixed; only the agent changes — that's what makes the improvement measurable. A held-out split prevents overfitting."
- "The agent runs on a local open model; paid calls are a sampled Haiku judge — total cost under a dollar."
- "It's offline evaluation on synthetic + golden data, not live production monitoring." (accurate, no overclaiming)

---

## 10. Final deliverables (end of Day 7)

Working application · Streamlit UI · evaluation system · self-improvement loop · reports · metrics dashboard · documentation · architecture diagram · public read-only demo link · resume line + interview talking points — all within budget.

---

## 11. Progress log

- **Day 0 (planning):** complete. Concept, architecture, cost-optimized plan, data strategy, deployment, and execution risks decided and documented (see the master guide, companion guide, and build-decisions docs). Stack and scope locked. Name: `eval-loop-agent`.
- **Day 1 (foundation + vertical slice):** complete. Scaffolded full tree (§5); `pyproject.toml` + `.venv` (deps installed); `config/settings.py` (pydantic-settings: `MODEL_PROVIDER`, paths, thresholds, sample sizes) + `config/logging.py`; Pydantic models in `storage/models.py` (`Task`, `AgentResponse`, `Trace`, `CriterionScore`, `EvalResult`, `Experiment`); `agents/llm.py` (Ollama default + Haiku w/ `cache_control`, retry/backoff); `agents/state.py` + `agents/graph.py` (single `generate` node); `storage/db.py` (5 tables, idempotent, sole SQLite module); minimal `evaluators/checks.py` (non_empty, length); `main.py` CLI (`--hello`, `run`). Tests: `test_models.py`, `test_db.py` (7 passing). **Verified:** ruff clean; `python main.py --hello` makes a live Ollama completion; `python main.py run "..."` writes task/response/trace/eval rows to `data/traces.db`.
- **Day 2 (first working agent — grounded response + tracing):** complete. KB domain = **AI evaluation & agent engineering**; authored 5 `datasets/kb/*.md` docs (RAG, LLM-as-judge, evaluation metrics, feedback loops, agent design) with `##`-section boundaries. `agents/tools.py` — pure, deterministic, embedding-free retriever (`load_kb` splits docs into section chunks; `retrieve` ranks by stopword-filtered token overlap, stable tie-break, returns top-k chunks + unique `doc_ids` + concatenated `context`; empty result on no overlap). `agents/prompts.py` — versioned `GENERATION_PROMPT_VERSION="v1"`, grounding system prompt + `build_generation_prompt` (answer from context only, admit when insufficient). `agents/graph.py` — now a two-node pipeline `retrieve → generate → END`; each node emits a `Trace` (`retrieve` payload carries query + doc_ids). `config/settings.py` — added `retrieval_top_k` (default 3). `main.py run` prints retrieved doc ids. Tests: `tests/test_graph.py` (4 retrieval unit tests + prompt-builder + graph wiring with model call monkeypatched — fully offline). Built test-first (TDD: red→green). **Verified:** 13 tests pass; ruff clean; 3 live sample tasks each produced KB-grounded answers with correct `doc_ids` (incl. multi-doc retrieval) and persisted `retrieve`+`generate` traces to `data/traces.db`. No storage/schema changes needed — Day 1 models were already retrieval-ready.
- **Next:** Day 3 — evaluation framework: golden + frozen synthetic datasets, deterministic grounding/coverage checks (`evaluators/checks.py`), rubric + sampled Haiku judge (`rubric.py`/`judge.py`), `runner.py`, and judge-vs-golden agreement validation.
