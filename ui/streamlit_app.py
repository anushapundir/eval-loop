"""UI layer — Streamlit demo + results explorer (CLAUDE.md §4/§6).

Two run modes, one app:

* **Live** (local): the "Run a task" form calls the FastAPI ``/run`` endpoint over
  HTTP (``httpx``) and renders v1 → feedback → v2 with metrics. Requires the API
  (``uvicorn app.main:app``) and a local Ollama.
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


def _render_eval_table(title: str, eval_result: dict) -> None:
    """Show a response's per-criterion scores (deterministic + judge) as a table."""
    rows = [
        {"criterion": s["name"], "score": round(s["score"], 3)}
        for s in (*eval_result.get("deterministic", []), *eval_result.get("judge", []))
    ]
    st.caption(title)
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_run_result(body: dict, threshold: float) -> None:
    """Render the v1 → feedback → v2 story plus the headline metrics."""
    v1, v2 = body["v1"], body["v2"]
    v1_eval, v2_eval = body["v1_eval"], body["v2_eval"]

    m1, m2, m3 = st.columns(3)
    m1.metric("v1 overall", f"{v1_eval['overall_score']:.3f}")
    m2.metric("v2 overall", f"{v2_eval['overall_score']:.3f}",
              delta=f"{body['improvement_delta']:+.3f}")
    m3.metric("retrieved docs", ", ".join(body["retrieved_doc_ids"]) or "(none)")

    st.caption(f"Pass threshold {threshold} · "
               f"{'judged (Haiku)' if body['judged'] else 'deterministic checks only'}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Response v1")
        st.write(v1["text"])
        _render_eval_table("v1 scores", v1_eval)
    with c2:
        st.subheader("Feedback")
        if body["revised"]:
            st.info(body["feedback"] or "(no feedback)")
        else:
            st.success("v1 already passed — carried forward unchanged as v2.")
    with c3:
        st.subheader("Response v2")
        st.write(v2["text"])
        _render_eval_table("v2 scores", v2_eval)

    with st.expander("Trace timeline"):
        st.dataframe(body["traces"], hide_index=True, use_container_width=True)


_STAGE_LABELS = {
    "retrieve": ("🔍 Retrieval · keyword match", "FREE"),
    "evaluate_v1": ("📊 Deterministic checks", "FREE"),
    "evaluate_v2": ("📊 Deterministic checks", "FREE"),
    "feedback": ("🛠 Rule-based critic", "FREE"),
    "carry_forward": ("➡ Carried forward (v1 passed)", "FREE"),
}


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


def _stream_run(settings, prompt: str, provider: str, do_judge: bool) -> None:
    """Open the SSE stream and render each pipeline stage as it arrives."""
    api = _api_base()
    payload = {"prompt": prompt, "provider": provider, "do_judge": do_judge}
    total_latency = 0.0
    paid_calls = 0

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
                event = json.loads(line[len("data: "):])
                kind = event.get("type")
                if kind == "stage":
                    title, cost = _stage_label(event["step"], event.get("provider"))
                    lat = event.get("latency_ms") or 0.0
                    total_latency += lat
                    if cost == "PAID":
                        paid_calls += 1
                    with st.status(f"{title} · {lat:.0f} ms · {cost}",
                                   state="complete", expanded=False):
                        _render_stage_payload(event["step"], event.get("payload", {}))
                elif kind == "judged":
                    st.caption(f"Judged (Haiku): v1 {event['v1_overall']:.3f} → "
                               f"v2 {event['v2_overall']:.3f}")
                elif kind == "done":
                    st.success(
                        f"Done · delta (v2−v1) {event['improvement_delta']:+.3f} · "
                        f"{'revised' if event['revised'] else 'carried forward unchanged'}"
                    )
                    st.caption(f"Total stage latency ≈ {total_latency:.0f} ms · "
                               f"paid model calls: {paid_calls}")
                elif kind == "error":
                    st.error(event.get("detail", "The run failed."))
                    return
    except httpx.ConnectError:
        st.error(f"Could not reach the API at {api}. Start it with "
                 "`uvicorn app.main:app --port 8000`.")
    except httpx.HTTPError as exc:
        st.error(f"Stream failed: {exc}")


def _run_section(settings) -> None:
    """The interactive 'Run a task' form (live mode only) → streaming /run/stream."""
    st.subheader("Run the self-improvement loop")
    st.caption("Submit a task → watch the agent retrieve, generate v1, critique, "
               "and revise to v2 — stage by stage, in real time.")

    with st.form("run_form"):
        prompt = st.text_area(
            "Task", placeholder="e.g. What is LLM-as-a-judge and why validate it?", height=120
        )
        provider_label = st.selectbox(
            "Model", ["Local · qwen2.5:7b (free)", "Claude Haiku 4.5 (paid — uses API budget)"]
        )
        do_judge = st.checkbox("Also run the Haiku judge after the loop (costs API budget)",
                               value=False)
        submitted = st.form_submit_button("Run", type="primary")

    if not submitted:
        return
    if not prompt.strip():
        st.warning("Please enter a task.")
        return

    provider = "haiku" if provider_label.startswith("Claude") else "ollama"
    _stream_run(settings, prompt, provider, do_judge)


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

    labels = {f"{e.name} · N={e.n_tasks} · {e.id[:8]}": e for e in experiments}
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

    st.set_page_config(page_title="eval-loop-agent", layout="wide")
    st.title("eval-loop-agent")
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
