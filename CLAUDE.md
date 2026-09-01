# CLAUDE.md — eval-loop

**Project:** `eval-loop` — a Self-Improving Agent with an Evaluation-Driven Feedback Loop.

This file is the **single source of truth** for Claude Code. Build the project from this document. When anything conflicts, this file wins. Do not expand scope, swap the stack, or skip the cost rules without explicit human approval. Workflow is **Path A: propose → human understands → human approves → execute.**

---

## 1. What this project is

`eval-loop` is an AI agent that improves its own output through a closed evaluation loop. Given a task, it generates a response, evaluates that response against a rubric, generates structured feedback, revises the response, and compares the versions — while an evaluation framework measures quality across a fixed dataset and tracks **measurable improvement** over iterations.

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
eval-loop/
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
- `docs/` — architecture and diagrams for reviewers.

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

- **Objective:** polish to production quality and complete the documentation.
- **Concepts:** production engineering practices, documentation, reproducibility, communicating results.
- **Features:** refactor for clarity; complete docstrings/type hints; finish tests; `docs/architecture.md` + diagram; final `README.md` (setup, run, results, screenshots, demo link); final cost check.
- **Files created:** `docs/architecture.md`, `docs/diagram.png`, finalized `README.md`, completed `tests/`.
- **Architecture changes:** none — hardening only.
- **Deliverables:** complete, documented, demo-ready repo with public link, results, and diagram.
- **Validation:** a fresh clone runs from the README; tests pass; demo link live; total spend under budget.
- **Success criteria:** a reviewer can understand the project in minutes, run it, and see measurable improvement; you can defend every part.
- **Expected output:** a documented, reproducible, demo-ready project.

---

## 9. Final deliverables (end of Day 7)

Working application · Streamlit UI · evaluation system · self-improvement loop · reports · metrics dashboard · documentation · architecture diagram · public read-only demo link — all within budget.

---

## 10. Progress log

- **Day 0 (planning):** complete. Concept, architecture, cost-optimized plan, data strategy, deployment, and execution risks decided and documented (see the master guide, companion guide, and build-decisions docs). Stack and scope locked. Name: `eval-loop`.
- **Day 1 (foundation + vertical slice):** complete. Scaffolded full tree (§5); `pyproject.toml` + `.venv` (deps installed); `config/settings.py` (pydantic-settings: `MODEL_PROVIDER`, paths, thresholds, sample sizes) + `config/logging.py`; Pydantic models in `storage/models.py` (`Task`, `AgentResponse`, `Trace`, `CriterionScore`, `EvalResult`, `Experiment`); `agents/llm.py` (Ollama default + Haiku w/ `cache_control`, retry/backoff); `agents/state.py` + `agents/graph.py` (single `generate` node); `storage/db.py` (5 tables, idempotent, sole SQLite module); minimal `evaluators/checks.py` (non_empty, length); `main.py` CLI (`--hello`, `run`). Tests: `test_models.py`, `test_db.py` (7 passing). **Verified:** ruff clean; `python main.py --hello` makes a live Ollama completion; `python main.py run "..."` writes task/response/trace/eval rows to `data/traces.db`.
- **Day 2 (first working agent — grounded response + tracing):** complete. KB domain = **AI evaluation & agent engineering**; authored 5 `datasets/kb/*.md` docs (RAG, LLM-as-judge, evaluation metrics, feedback loops, agent design) with `##`-section boundaries. `agents/tools.py` — pure, deterministic, embedding-free retriever (`load_kb` splits docs into section chunks; `retrieve` ranks by stopword-filtered token overlap, stable tie-break, returns top-k chunks + unique `doc_ids` + concatenated `context`; empty result on no overlap). `agents/prompts.py` — versioned `GENERATION_PROMPT_VERSION="v1"`, grounding system prompt + `build_generation_prompt` (answer from context only, admit when insufficient). `agents/graph.py` — now a two-node pipeline `retrieve → generate → END`; each node emits a `Trace` (`retrieve` payload carries query + doc_ids). `config/settings.py` — added `retrieval_top_k` (default 3). `main.py run` prints retrieved doc ids. Tests: `tests/test_graph.py` (4 retrieval unit tests + prompt-builder + graph wiring with model call monkeypatched — fully offline). Built test-first (TDD: red→green). **Verified:** 13 tests pass; ruff clean; 3 live sample tasks each produced KB-grounded answers with correct `doc_ids` (incl. multi-doc retrieval) and persisted `retrieve`+`generate` traces to `data/traces.db`. No storage/schema changes needed — Day 1 models were already retrieval-ready.
- **Day 3 (evaluation framework — the measuring instrument):** complete. **Datasets:** `datasets/loader.py` (`load_dataset`, pure JSONL→`Task`, optional `split` filter); `datasets/golden/golden.jsonl` (23 human-verifiable records across all 5 KB topics + edge cases — out-of-KB, multi-doc; `split` dev=17/test=6); `datasets/generate.py` (deterministic, free, template×topic synthetic generator + documented spec) → frozen `datasets/synthetic/synthetic.jsonl` (100 distinct, unlabeled). **Deterministic checks** (`evaluators/checks.py`): added `grounding_check` (response↔context token overlap; 1.0 + "not applicable" when no context, so honest "don't know" isn't penalized) and `coverage_check` (token-subset key-point match; 1.0 when none) composed by `run_deterministic_checks`; reuses the retriever's tokenizer (promoted `agents/tools.py:_tokenize`→public `tokenize`). **Rubric** (`evaluators/rubric.py`): `RUBRIC` (correctness/completeness/clarity) as data with high/low anchors + `RUBRIC_VERSION="v1"`, `render_rubric()`. **Judge** (`evaluators/judge.py`): `judge_response` forces `provider="haiku"` + `cache_system=True`, strict-JSON parse tolerant of prose/code-fences, parse-retry then graceful degrade to `[]` (`judged=False`) — never crashes a batch; clamps scores to [0,1]. Versioned judge prompt added to `agents/prompts.py` (`JUDGE_PROMPT_VERSION`, `JUDGE_SYSTEM`, `build_judge_prompt`). **Runner** (`evaluators/runner.py`): `select_judge_indices` (seeded, reproducible sampling), `evaluate_response` (overall = det mean, or 0.5·det+0.5·judge when judged), `run_dataset` (responder-injected, persists Task/Response/EvalResult via `storage/db.py`). **Judge validation** (`evaluators/validate_judge.py`): good-vs-bad ranking (reference vs off-topic/truncated probes) → `ranking_accuracy` + `good_pass_rate`, reported with honest N. **CLI:** `main.py evals --dataset {golden,synthetic} [--judge-sample-rate R] [--no-judge]` (writes `Experiment` + EvalResults) and `validate-judge --limit N`; `run` upgraded to the full deterministic suite. Built test-first (TDD red→green). **Verified:** 44 tests pass (added `test_datasets`, `test_checks`, `test_judge`, `test_runner`, `test_validate_judge`, `test_cli`); ruff clean; live `python main.py evals --dataset golden --no-judge` over local Ollama scored 23/23, **mean overall 0.850, pass rate 19/23**, persisting 23 EvalResults (4 deterministic criteria each) + an `Experiment` (`mean_v1=0.85`, `mean_v2=None`) to `data/traces.db`; the out-of-KB task correctly retrieved nothing and gave a short honest answer. No schema/settings changes needed — Day 1 models + eval knobs were already Day-3-ready. **Judge validated live:** with `ANTHROPIC_API_KEY` set, `python main.py validate-judge --limit 5` (15 Haiku calls, cached rubric) returned **ranking accuracy 1.00, good pass rate 1.00, mean good 0.93 vs mean bad 0.30** — the judge reliably ranks reference answers above degraded (off-topic/truncated) variants (directional at N=5). **Pending human action:** verify the 23 drafted golden labels before relying on them; a full sampled-judge `evals --dataset golden` run has not yet been done live (cost-gated).
- **Day 4 (feedback + self-improvement loop — the loop closes):** complete. **Revision prompt** (`agents/prompts.py`): versioned `REVISION_PROMPT_VERSION="v1"`, `REVISION_SYSTEM` + `build_revision_prompt` (apply feedback, stay grounded in the same context; deliberately takes only previous-answer + feedback, **never the golden `expected`** → no label leakage). **Feedback generator** (`feedback/generator.py`, the "critic" role, distinct from doer/judge): pure, deterministic, **free** `generate_feedback(eval_result, task, response_text)` → `Feedback`(items + rendered text); maps low deterministic scores to actionable instructions and, crucially, names the **exact missing key points** (reuses `tokenize`, one tokenizer no drift); empty/`is_actionable=False` when nothing needs fixing. New Pydantic `FeedbackItem`/`Feedback` models in `storage/models.py` (additive; no schema migration). **Revise step** (`feedback/improve.py`): `revise(...)` calls `agents/llm.generate` on the **default free provider** (never forced to Haiku — cost rule §2). **Graph** (`agents/graph.py`): extended to `retrieve → generate → evaluate_v1 →` *(conditional `should_revise`)* `→ feedback → revise → evaluate_v2 → END`, else `→ carry_forward → END`; in-loop eval is **deterministic-only (free)** — the paid judge never runs inside the loop. Single revision (v1→v2, matching the two state slots); `should_revise` short-circuits to carry-forward when v1 already passes **or** `iteration ≥ max_iterations` (guaranteed termination; v2 always exists so the dataset delta is an honest value, never missing). **Runner** (`evaluators/runner.py`): added `run_improvement(tasks, loop_runner, …) → ImprovementResults`; reuses `evaluate_response` + `select_judge_indices` (**same sampled indices judged for both v1 and v2** → fair comparison), persists both responses/traces/EvalResults, aggregates `mean_v1`/`mean_v2`/`improvement_delta`/`n_judged`. **CLI:** `run` upgraded to display the full **v1 → feedback → v2** loop with both scores + delta; `evals` gained `--loop` (opt-in; Day-3 v1-only path preserved by default) and `--split {dev,test}`, writing the `Experiment` row with `mean_v1`/`mean_v2`/`improvement_delta`. Built test-first (TDD red→green). **Verified:** 55 tests pass (added `test_feedback`, `test_improve`; extended `test_graph` loop-paths incl. carry-forward + max_iterations=0, `test_runner` same-sample-judge + delta math, `test_cli` `--loop`/`--split`); ruff clean. **Live (local Ollama, free, `--no-judge`):** held-out `test` split (N=6) `mean_v1=0.843 → mean_v2=0.873`, **delta +0.030** (1/6 revised upward, **0 regressions**, 5/6 carried forward); `dev` split (N=17) `mean_v1=0.848 → mean_v2=0.878`, **delta +0.030** (3/17 revised upward, **0 regressions**, 14/17 carried forward) — consistent positive, regression-free at larger N. Each persisted an `Experiment` (`mean_v1`/`mean_v2`/`improvement_delta` all set) + version-tagged EvalResults (e.g. 12 for the test run = 6 v1 + 6 v2) to `data/traces.db`. Honest framing: directional at small N (CLAUDE.md §7), positive and regression-free; most v1 answers already pass on this KB, so the loop engages only the weak ones — the framework correctly *measures* that rather than inflating it (no best-of gate; judge never in the loop). **Pending human action:** verify the golden labels (carried over from Day 3); a sampled-judge `--loop` run is cost-gated and not yet done live.
- **Day 5 (experiment tracking, metrics, comparison, reporting — the dashboard data):** complete. **Zero model calls — pure reads from SQLite + pandas + matplotlib; cost $0.** **Metrics** (`reports/metrics.py`, pure over the storage models, no DB/render coupling): `summarize_experiment` → `ExperimentSummary` (recomputes mean_v1/mean_v2/delta + v1/v2 pass-rates from the stored eval rows, self-consistent even if re-scored; delta None when no v2); `per_criterion_means` (pandas pivot of deterministic+judge criteria → v1/v2 columns; tolerates empty judge lists from `--no-judge` runs); `compare_versions` → `VersionComparison` (overall + per-criterion deltas, only for criteria in both versions); `experiment_trend` (chronological DataFrame across experiments for the trend line); `check_regression` → `RegressionResult` (flags `current < baseline − tolerance`; no baseline → passes). **Failure-mode clustering** (`evaluators/analysis.py`, its §5 home — deterministic, free): `cluster_failures` → `FailureAnalysis` attributes each failing result's below-threshold criteria to named modes (`low_grounding`/`missing_coverage`/`length_violation`/`incorrect`/…; `other` when overall fails but no single criterion does), counts sorted desc. **Charts** (`reports/charts.py`, `matplotlib.use("Agg")` before pyplot for headless Windows): four pure renderers (`chart_score_trend`, `chart_v1_vs_v2`, `chart_per_criterion`, `chart_failure_modes`) each take aggregated data + write one PNG; empty inputs still emit a labeled placeholder so artifacts always exist; `render_all` orchestrates all four (failure modes computed over v1 = what fails before revision). **Settings:** added `regression_tolerance` (default 0.0, strict; §6 no-hardcode). **CLI:** single `report [--experiment-id ID] [--gate]` command (`cmd_report`) — aggregates the focal (default newest) experiment, writes `report_summary.json` + `per_criterion.csv` + 4 charts to `reports/output/`, prints the summary; `--gate` compares the headline mean vs the prior experiment and exits non-zero on regression. Built test-first (TDD red→green). **Verified:** 82 tests pass (added `test_metrics` 13, `test_analysis` 5, `test_charts` 7, `test_cli` +2; all offline, `tmp_path` for output isolation); ruff clean. **Live (free, reads `data/traces.db`):** `python main.py report --gate` on the newest `loop-golden` (N=6) printed **mean_v1 0.843 → mean_v2 0.878, delta +0.035, v1 pass 0.833 → v2 pass 1.000**, failure modes `low_grounding=1, missing_coverage=1` (1/6 v1 failed), wrote all four PNGs + JSON + CSV, and the gate passed (OK vs equal baseline). Per-criterion breakdown shows the v2 gain is driven by **coverage 0.556→0.722** (grounding dips 0.816→0.789) — an honest, interpretable signal, not an inflated headline. `.gitignore` already scopes `reports/output/*` to ignore generated artifacts except the committed `demo/` set (Day 6). No storage schema changes — Day 1 models + the existing `db.list_experiments`/`list_eval_results` read API were already Day-5-ready. **Pending human action:** verify the golden labels (carried from Day 3); a sampled-judge run + report remains cost-gated.
- **Day 6 (FastAPI app + Streamlit UI + zero-cost demo — the two missing layers):** complete. **Application layer** (`app/main.py`, thin FastAPI boundary so the UI never imports the agent): `GET /health` (liveness), `POST /run` (`RunRequest{prompt, do_judge=False}` → `RunResponse{task, v1, v2, v1_eval, v2_eval, feedback, revised, improvement_delta, retrieved_doc_ids, judged, traces}`) mirroring `cmd_run` — builds the graph, invokes the loop, persists task/responses/traces/evals, and **deterministic-only by default**; `do_judge` re-scores v1+v2 with the sampled Haiku judge *after* the loop (judge never in the loop, §2/§7). Agent failure → HTTP 503 (not a 500 stack trace). `GET /results` lists stored experiments via `summarize_experiment`. Lifespan hook runs `ensure_dirs()`+`init_db()` on startup. API I/O is all Pydantic (§6). **UI layer** (`ui/streamlit_app.py`, two modes, one app): **live** mode calls FastAPI `/run` over `httpx` and renders v1 | feedback | v2 columns + metrics (graceful `st.error` on `ConnectError`/503); **demo** mode (`EVAL_LOOP_DEMO_MODE=1`) hides the run form and serves only the committed read-only snapshot — **zero live API calls / no Ollama** for Streamlit Community Cloud. Shared read-only **results explorer** (storage = integration boundary, §4) lists experiments + per-criterion table + the four charts (precomputed PNGs in demo mode; on-demand `render_all` in live mode). **Demo build** (`main.py demo` → `cmd_demo`): free, deterministic-only; rebuilds a fresh `data/demo_results.db` by running the golden `dev` then held-out `test` splits through the loop (two experiments → a two-point trend), then writes `reports/output/demo/` (4 PNGs + JSON + CSV) via the new shared `_write_report_artifacts` helper (also refactored into `cmd_report`, no behavior change). `.gitignore` already force-adds `data/demo_results.db` + `reports/output/demo/` (verified via `git check-ignore`). Built test-first (TDD red→green). **Verified:** **88 tests pass** (added `tests/test_app.py` — TestClient over `/health`, `/run` with a monkeypatched graph + stubbed DB writes (fully offline), 503 path, blank-prompt 422, `/results` summaries; `test_cli` +1 for the `demo` parser); ruff clean. **Live (local Ollama, free):** `uvicorn app.main:app` → `/health` 200, `/results` returned stored experiment summaries, `POST /run` on a real task retrieved `['evaluation-metrics','llm-as-judge']`, scored v1=0.836 (passed→carried forward), full trace `retrieve→generate→evaluate→carry_forward`. `python main.py demo` built `demo_results.db` (2 experiments: dev N=17 **delta +0.042**, test N=6 **delta +0.032**, 46 eval rows) + all four committed PNGs/JSON/CSV. README updated (CLI table, two-process run, demo-mode + deploy notes). No storage/schema changes — Days 1–5 layers were consumed unchanged. **Pending human action:** verify the golden labels (carried from Day 3); deploy `ui/streamlit_app.py` to Streamlit Community Cloud with `EVAL_LOOP_DEMO_MODE=1` to publish the public link; a sampled-judge run remains cost-gated.
- **Next:** Day 7 — refactor/polish, complete docstrings + type hints, finish tests, `docs/architecture.md` + diagram, final `README.md` (screenshots + live demo link), and the final cost check.
