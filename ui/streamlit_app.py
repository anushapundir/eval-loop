"""UI layer — Streamlit demo + results explorer (CLAUDE.md §4/§6).

Two run modes, one app:

* **Live** (local): the "Run a task" form streams the pipeline stage-by-stage
  from the FastAPI ``/run/stream`` SSE endpoint and renders each stage
  (retrieve → generate → evaluate → feedback → revise → re-evaluate) as it
  completes, with a model provider toggle (local Ollama / Claude Haiku).
  Requires ``uvicorn app.main:app`` and a local Ollama (unless Haiku is selected).
* **Demo** (deployed): set ``EVAL_LOOP_DEMO_MODE=1``. The run form is hidden and
  every read is served from the committed, read-only ``demo_results.db`` plus the
  precomputed charts in ``reports/output/demo/`` — **zero live API calls**, so it
  runs free on Streamlit Community Cloud with no Ollama and no API key.

The results explorer (read-only reporting view) reads the storage layer directly
— storage is the integration boundary (§4) — and renders the four Day 5 charts.
Honest framing per §7/§9 is kept visible in the copy: offline eval, frozen
weights, small N.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import streamlit as st

from config.settings import get_settings
from reports.charts import render_all
from reports.metrics import per_criterion_means, summarize_experiment
from storage import db
from storage.models import Experiment

_TRUTHY = {"1", "true", "yes", "on"}


def _is_demo_mode() -> bool:
    """Demo mode is on when EVAL_LOOP_DEMO_MODE is truthy (deployed, read-only)."""
    return os.getenv("EVAL_LOOP_DEMO_MODE", "").strip().lower() in _TRUTHY


def _api_base() -> str:
    """Base URL of the FastAPI server (overridable for non-default local ports)."""
    return os.getenv("EVAL_LOOP_API_URL", "http://localhost:8000").rstrip("/")


def _explorer_db_path(settings) -> Path:
    """Which SQLite DB the explorer reads: the demo snapshot or the live DB."""
    return settings.demo_db_path if _is_demo_mode() else settings.db_path


# --- rendering helpers ------------------------------------------------------


_STAGE_LABELS = {
    "retrieve": ("🔍 Retrieval · keyword match", "FREE"),
    "evaluate_v1": ("📊 Deterministic checks", "FREE"),
    "evaluate_v2": ("📊 Deterministic checks", "FREE"),
    "feedback": ("🛠 Rule-based critic", "FREE"),
    "carry_forward": ("➡ Carried forward (v1 passed)", "FREE"),
}


# Curated example tasks (real golden records) whose v1 answer typically misses a
# key point, so at a strict quality bar the loop genuinely revises and improves —
# the difference a free-form task can't reliably show. Ordered: default loads
# ready to run. `key_points` mirror datasets/golden/golden.jsonl.
EXAMPLE_TASKS: dict[str, dict | None] = {
    "⭐ Why score each criterion separately? (revises)": {
        "prompt": "Why score each criterion separately instead of one number?",
        "key_points": ["where", "weak", "overall score"],
    },
    "How do metrics + feedback loops improve an agent? (revises)": {
        "prompt": "How do evaluation metrics and feedback loops work together to improve an agent?",
        "key_points": ["scores", "feedback", "improvement delta"],
    },
    "Custom task…": None,
}
_DEFAULT_EXAMPLE = next(iter(EXAMPLE_TASKS))


def _render_judge_panel(event: dict) -> None:
    """Render the LLM judge's per-criterion verdict (v1 vs v2) with justifications.

    This is what the deterministic checks can't measure: subjective quality scored
    against the rubric. Shown only when the judge ran (the 'judged' SSE event).
    """
    st.markdown("#### 🧑‍⚖️ LLM judge (Haiku) — subjective quality vs the rubric")
    st.caption("Scores correctness · completeness · clarity against the rubric, using "
               "the retrieved context — not a reference answer. This is the quality "
               "the free deterministic checks cannot capture.")

    criteria = event.get("criteria") or []
    if not criteria:
        st.caption(f"Judge overall: v1 {event['v1_overall']:.3f} → v2 "
                   f"{event['v2_overall']:.3f} (per-criterion detail unavailable)")
        return

    rows = []
    for c in criteria:
        v1, v2 = c.get("v1"), c.get("v2")
        delta = round(v2 - v1, 3) if (v1 is not None and v2 is not None) else None
        rows.append({"criterion": c["name"], "v1": v1, "v2": v2, "Δ (v2−v1)": delta})
    st.dataframe(rows, hide_index=True, use_container_width=True)
    st.caption(f"Judge overall (blended into the score): v1 {event['v1_overall']:.3f} → "
               f"v2 {event['v2_overall']:.3f}")

    for c in criteria:
        with st.expander(f"Judge's reasoning — {c['name']}"):
            st.markdown(f"**v1:** {c.get('v1_justification') or '—'}")
            st.markdown(f"**v2:** {c.get('v2_justification') or '—'}")


def _render_comparison(v1_payload: dict, v2_payload: dict) -> None:
    """Consolidated v1→v2 scoreboard for the deterministic checks + overall.

    Puts every criterion's v1/v2/delta in one place so the specific gain the loop
    made (typically coverage, when the critic names a missing key-point) is obvious
    at a glance — the overall delta alone is small because two checks are always maxed.
    """
    st.markdown("#### 📊 v1 → v2 comparison (deterministic checks)")
    v1s = {s["name"]: s["score"] for s in (v1_payload.get("scores") or [])}
    v2s = {s["name"]: s["score"] for s in (v2_payload.get("scores") or [])}

    rows = []
    for name in (list(v1s) or list(v2s)):
        a, b = v1s.get(name), v2s.get(name)
        delta = round(b - a, 3) if (a is not None and b is not None) else None
        rows.append({"criterion": name, "v1": a, "v2": b, "Δ (v2−v1)": delta})

    o1, o2 = v1_payload.get("overall"), v2_payload.get("overall")
    rows.append({
        "criterion": "overall",
        "v1": o1, "v2": o2,
        "Δ (v2−v1)": round(o2 - o1, 3) if (o1 is not None and o2 is not None) else None,
    })
    st.dataframe(rows, hide_index=True, use_container_width=True)

    gains = [(r["Δ (v2−v1)"], r["criterion"]) for r in rows
             if r["criterion"] != "overall" and r["Δ (v2−v1)"] and r["Δ (v2−v1)"] > 0]
    if gains:
        best_delta, best_name = max(gains)
        st.caption(f"Biggest gain: **{best_name}** ({best_delta:+.3f}) — the critic named "
                   "this gap and the revision closed it.")
    elif o1 is not None and o2 is not None and o2 == o1:
        st.caption("v1 already cleared the bar, so it carried forward unchanged (delta 0).")


def _stage_label(step: str, provider: str | None) -> tuple[str, str]:
    """Return (title, cost_chip) for a pipeline stage card."""
    if step in ("generate", "revise"):
        if provider == "haiku":
            return ("✦ Claude Haiku 4.5", "PAID")
        return ("🦙 Local · qwen2.5:7b", "FREE")
    return _STAGE_LABELS.get(step, (step, "FREE"))


def _render_stage_payload(step: str, payload: dict) -> None:
    """Render the intermediate output for one stage inside its card."""
    if step == "retrieve":
        docs = payload.get("doc_ids") or []
        st.write("**Retrieved docs:** " + (", ".join(docs) if docs else "(none matched)"))
    elif step in ("generate", "revise"):
        st.write(payload.get("text", ""))
    elif step in ("evaluate_v1", "evaluate_v2", "carry_forward"):
        scores = payload.get("scores")
        if scores:
            st.dataframe(scores, hide_index=True, use_container_width=True)
        if "overall" in payload:
            verdict = "PASS" if payload.get("passed") else "FAIL"
            st.caption(f"overall {payload['overall']:.3f} → {verdict}")
    elif step == "feedback":
        st.info(payload.get("text") or "(no feedback)")


def _stream_run(settings, prompt: str, provider: str, do_judge: bool,
                key_points: list[str], pass_threshold: float) -> None:
    """Open the SSE stream and render each pipeline stage as it arrives."""
    api = _api_base()
    payload = {
        "prompt": prompt,
        "provider": provider,
        "do_judge": do_judge,
        "key_points": key_points,
        "pass_threshold": pass_threshold,
    }
    total_latency = 0.0
    paid_calls = 0
    stream_complete = False
    v1_eval_payload: dict | None = None
    v2_eval_payload: dict | None = None

    st.subheader("Pipeline (live)")
    try:
        with httpx.stream("POST", f"{api}/run/stream", json=payload, timeout=300.0) as resp:
            if resp.status_code != 200:
                resp.read()
                st.error(f"API returned {resp.status_code}: {resp.text}")
                return
            for line in resp.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                raw = line[len("data: "):]
                if not raw:
                    continue
                event = json.loads(raw)
                kind = event.get("type")
                if kind == "stage":
                    title, cost = _stage_label(event["step"], event.get("provider"))
                    lat = event.get("latency_ms") or 0.0
                    total_latency += lat
                    if cost == "PAID":
                        paid_calls += 1
                    payload = event.get("payload", {})
                    if event["step"] == "evaluate_v1":
                        v1_eval_payload = payload
                    elif event["step"] in ("evaluate_v2", "carry_forward"):
                        v2_eval_payload = payload
                    with st.status(f"{title} · {lat:.0f} ms · {cost}",
                                   state="complete", expanded=False):
                        _render_stage_payload(event["step"], payload)
                elif kind == "judged":
                    paid_calls += 2
                    _render_judge_panel(event)
                elif kind == "done":
                    stream_complete = True
                    st.success(
                        f"Done · delta (v2−v1) {event['improvement_delta']:+.3f} · "
                        f"{'revised' if event['revised'] else 'carried forward unchanged'}"
                    )
                    if v1_eval_payload is not None and v2_eval_payload is not None:
                        _render_comparison(v1_eval_payload, v2_eval_payload)
                    st.caption(f"Total stage latency ≈ {total_latency:.0f} ms · "
                               f"paid model calls: {paid_calls}")
                elif kind == "error":
                    st.error(event.get("detail", "The run failed."))
                    return
            if not stream_complete:
                st.warning("The pipeline stream ended without a final result. The agent "
                           "may still be running, or the connection was interrupted.")
    except httpx.ConnectError:
        st.error(f"Could not reach the API at {api}. Start it with "
                 "`uvicorn app.main:app --port 8000`.")
    except httpx.HTTPError as exc:
        st.error(f"Stream failed: {exc}")


def _on_example_change() -> None:
    """Sync the task text box to the picked example (callback runs before rerender)."""
    example = EXAMPLE_TASKS[st.session_state.example_choice]
    if example is not None:
        st.session_state.task_prompt = example["prompt"]


def _run_section(settings) -> None:
    """The interactive 'Run a task' form (live mode only) → streaming /run/stream."""
    st.subheader("Run the self-improvement loop")
    st.caption("Submit a task → watch the agent retrieve, generate v1, critique, "
               "and revise to v2 — stage by stage, in real time.")

    # Seed session state so the page loads with the default example ready to run.
    if "example_choice" not in st.session_state:
        st.session_state.example_choice = _DEFAULT_EXAMPLE
    if "task_prompt" not in st.session_state:
        seed = EXAMPLE_TASKS[_DEFAULT_EXAMPLE]
        st.session_state.task_prompt = seed["prompt"] if seed else ""

    st.selectbox(
        "Example task", list(EXAMPLE_TASKS), key="example_choice",
        on_change=_on_example_change,
        help="Example tasks carry required key-points, so at a strict quality bar "
             "the loop actually revises. Pick “Custom task…” to type your own.",
    )
    example = EXAMPLE_TASKS[st.session_state.example_choice]

    with st.form("run_form"):
        prompt = st.text_area("Task", key="task_prompt", height=120)
        col1, col2 = st.columns(2)
        provider_label = col1.selectbox(
            "Model", ["Local · qwen2.5:7b (free)", "Claude Haiku 4.5 (paid — uses API budget)"]
        )
        pass_threshold = col2.slider(
            "Quality bar (pass threshold)", 0.50, 1.00, 0.80, 0.05,
            help="Answers scoring below this get critiqued and revised. 0.80 sits "
                 "between a typical v1 and v2, so the example tasks fail v1, get "
                 "revised, and v2 clears the bar. (Grounding is a word-overlap proxy, "
                 "so realistic answers rarely exceed ~0.85 — 0.90 is effectively "
                 "unreachable.)",
        )
        do_judge = st.checkbox("Also run the Haiku judge after the loop "
                               "(costs API budget; shows the judge panel)", value=False)
        submitted = st.form_submit_button("Run", type="primary")

    if not submitted:
        return
    if not prompt.strip():
        st.warning("Please enter a task.")
        return

    # Key-points only apply to the *unedited* example prompt. If the task text was
    # retyped, run it as a plain custom task — otherwise we'd score the new answer
    # against key-points from a different question (a confusing false failure).
    is_example = example is not None and prompt.strip() == example["prompt"]
    key_points = example["key_points"] if is_example else []
    if is_example:
        st.caption("✓ Example task — the coverage check looks for these key-points: "
                   + ", ".join(f"`{k}`" for k in key_points))
    elif example is not None:
        st.info("You edited the example text, so this runs as a **custom task** — no "
                "key-point coverage check (pick “Custom task…” to hide this notice).")

    provider = "haiku" if provider_label.startswith("Claude") else "ollama"
    _stream_run(settings, prompt, provider, do_judge, key_points, pass_threshold)


def _load_experiments(settings) -> list[Experiment]:
    """Read experiments from the active DB (demo snapshot or live)."""
    path = _explorer_db_path(settings)
    if not path.exists():
        return []
    db.init_db(path)
    return db.list_experiments(path)


def _explorer_charts(exp: Experiment, settings) -> dict[str, Path]:
    """Resolve the four charts for an experiment.

    Demo mode serves the committed precomputed PNGs (no matplotlib at request
    time); live mode renders them on demand from the live DB.
    """
    if _is_demo_mode():
        demo_dir = settings.reports_output_dir / "demo"
        keys = ("score_trend", "v1_vs_v2", "per_criterion", "failure_modes")
        return {k: demo_dir / f"{k}.png" for k in keys}

    path = _explorer_db_path(settings)
    results = db.list_eval_results(exp.id, path)
    all_experiments = db.list_experiments(path)
    return render_all(exp, results, all_experiments,
                      out_dir=settings.reports_output_dir, settings=settings)


def _explorer_section(settings) -> None:
    """The read-only results explorer (both modes): summaries + charts."""
    st.subheader("Results explorer")
    experiments = _load_experiments(settings)
    if not experiments:
        st.info("No experiments yet. Run `python main.py evals --loop` (or `demo`) first.")
        return

    labels = {
        f"{e.name} · N={e.n_tasks} · {e.created_at:%Y-%m-%d %H:%M} · {e.id[:8]}": e
        for e in experiments
    }
    choice = st.selectbox("Experiment", list(labels.keys()))
    exp = labels[choice]

    path = _explorer_db_path(settings)
    results = db.list_eval_results(exp.id, path)
    summary = summarize_experiment(exp, results)

    c1, c2, c3 = st.columns(3)
    c1.metric("mean v1", f"{summary.mean_v1:.3f}" if summary.mean_v1 is not None else "n/a")
    c2.metric(
        "mean v2",
        f"{summary.mean_v2:.3f}" if summary.mean_v2 is not None else "n/a",
        delta=(f"{summary.improvement_delta:+.3f}"
               if summary.improvement_delta is not None else None),
    )
    c3.metric(
        "v2 pass rate",
        f"{summary.v2_pass_rate:.2f}" if summary.v2_pass_rate is not None else "n/a",
    )

    table = per_criterion_means(results)
    if not table.empty:
        st.caption("Per-criterion mean score (v1 vs v2)")
        st.dataframe(table, use_container_width=True)

    st.caption("Charts")
    charts = _explorer_charts(exp, settings)
    grid = st.columns(2)
    for i, (key, chart_path) in enumerate(charts.items()):
        with grid[i % 2]:
            if Path(chart_path).exists():
                st.image(str(chart_path), caption=key)
            else:
                st.warning(f"Missing chart: {key}")


def main() -> None:
    """Render the Streamlit app (run section + results explorer)."""
    settings = get_settings()
    demo = _is_demo_mode()

    st.set_page_config(page_title="eval-loop", layout="wide")
    st.title("eval-loop")
    st.caption("A self-improving agent with an evaluation-driven feedback loop. "
               "Offline eval on synthetic + golden data; model weights are frozen "
               "(program optimization, not retraining). Trends are directional at small N.")

    with st.sidebar:
        st.header("Mode")
        if demo:
            st.success("Demo (read-only)")
            st.caption("Serving precomputed results — zero live API calls.")
        else:
            st.info("Live (local)")
            st.caption(f"API: {_api_base()}")
        st.divider()
        st.caption("CLAUDE.md §2: agent runs on local Ollama (free); the Haiku judge "
                   "is sampled and opt-in.")

    if demo:
        _explorer_section(settings)
    else:
        run_tab, explorer_tab = st.tabs(["Run a task", "Results explorer"])
        with run_tab:
            _run_section(settings)
        with explorer_tab:
            _explorer_section(settings)


# A Streamlit script is executed top-to-bottom on every rerun (not imported),
# so render unconditionally.
main()
