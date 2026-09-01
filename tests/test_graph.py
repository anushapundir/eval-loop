"""Tests for retrieval (agents/tools.py) and graph wiring (agents/graph.py).

Retrieval is tested as a pure function against a tiny fixture KB. The graph is
tested with the model call monkeypatched, so it runs fully offline (no Ollama).
"""

from __future__ import annotations

from pathlib import Path

from agents import graph as graph_module
from agents.llm import Completion
from agents.prompts import GENERATION_PROMPT_VERSION, build_generation_prompt
from agents.state import AgentState
from agents.tools import KbChunk, load_kb, retrieve
from feedback import improve as improve_module
from storage.models import Task


def _patch_models(monkeypatch, *, context: str, v1_text: str, v2_text: str) -> None:
    """Make the graph offline: fixed retrieval context, scripted v1 and v2 text."""
    monkeypatch.setattr(
        graph_module, "load_kb", lambda _dir: [KbChunk(doc_id="kb", title="T", text=context)]
    )
    monkeypatch.setattr(
        graph_module,
        "generate",
        lambda prompt, system=None, **kw: Completion(
            text=v1_text, provider="ollama", model="test", latency_ms=1.0
        ),
    )
    monkeypatch.setattr(
        improve_module,
        "generate",
        lambda prompt, system=None, **kw: Completion(
            text=v2_text, provider="ollama", model="test", latency_ms=1.0
        ),
    )


def _write(tmp_path: Path, name: str, text: str) -> None:
    (tmp_path / name).write_text(text, encoding="utf-8")


def test_load_kb_splits_doc_into_section_chunks(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "doc.md",
        "# Title\n\nintro line\n\n## Section A\nalpha beta\n\n## Section B\ngamma delta\n",
    )
    chunks = load_kb(tmp_path)

    titles = {c.title for c in chunks}
    assert "Section A" in titles
    assert "Section B" in titles
    # doc_id is the filename stem for every chunk of the doc.
    assert {c.doc_id for c in chunks} == {"doc"}


def test_retrieve_ranks_most_relevant_chunk_first(tmp_path: Path) -> None:
    chunks = [
        KbChunk(doc_id="feedback", title="Loop", text="feedback loop critique revise reflexion"),
        KbChunk(doc_id="rag", title="Grounding", text="retrieval grounding documents context"),
    ]
    result = retrieve("how does a feedback loop critique and revise", chunks, k=1)

    assert result.doc_ids == ["feedback"]
    assert "feedback" in result.context.lower()


def test_retrieve_returns_empty_when_no_token_overlap() -> None:
    chunks = [KbChunk(doc_id="d1", title="A", text="alpha beta gamma")]
    result = retrieve("zzz qqq wwww", chunks, k=3)

    assert result.doc_ids == []
    assert result.context == ""


def test_retrieve_is_deterministic(tmp_path: Path) -> None:
    chunks = [
        KbChunk(doc_id="a", title="A", text="judge rubric bias validation golden"),
        KbChunk(doc_id="b", title="B", text="judge rubric bias validation golden"),
    ]
    first = retrieve("judge rubric bias", chunks, k=2)
    second = retrieve("judge rubric bias", chunks, k=2)
    assert first.doc_ids == second.doc_ids


def test_build_generation_prompt_includes_context_and_question() -> None:
    prompt = build_generation_prompt(question="what is RAG?", context="[rag] grounding")
    assert "what is RAG?" in prompt
    assert "[rag] grounding" in prompt
    assert GENERATION_PROMPT_VERSION == "v1"


def test_graph_loop_revises_when_v1_fails(monkeypatch) -> None:
    """A weak v1 (low grounding + coverage) triggers feedback → revise → v2."""
    _patch_models(
        monkeypatch,
        context="alpha beta gamma delta",
        v1_text="completely unrelated words here padding the length enough",
        v2_text="alpha beta gamma delta is the grounded improved answer",
    )
    task = Task(prompt="explain alpha", key_points=["alpha", "beta"])

    compiled = graph_module.build_graph()
    state = AgentState(**compiled.invoke(AgentState(task=task)))

    assert state.v1 is not None and state.v2 is not None
    assert state.v1.text != state.v2.text
    assert state.v2.text == "alpha beta gamma delta is the grounded improved answer"
    assert state.v2.version.value == "v2"
    assert state.v1_eval is not None and state.v2_eval is not None
    assert state.feedback  # non-empty actionable feedback
    steps = [t.step for t in state.traces]
    assert steps[:2] == ["retrieve", "generate"]
    assert "feedback" in steps and "revise" in steps
    assert "carry_forward" not in steps


def test_graph_carries_forward_when_v1_passes(monkeypatch) -> None:
    """A strong v1 short-circuits: no revise call, v2 carries v1 forward."""
    revise_called = {"n": 0}

    def fake_revise_generate(prompt, system=None, **kw):
        revise_called["n"] += 1
        return Completion(text="SHOULD NOT BE CALLED", provider="ollama", model="t", latency_ms=1.0)

    monkeypatch.setattr(
        graph_module,
        "load_kb",
        lambda _dir: [KbChunk(doc_id="kb", title="T", text="alpha beta gamma delta epsilon")],
    )
    monkeypatch.setattr(
        graph_module,
        "generate",
        lambda prompt, system=None, **kw: Completion(
            text="alpha beta gamma delta epsilon answer here",
            provider="ollama", model="test", latency_ms=1.0,
        ),
    )
    monkeypatch.setattr(improve_module, "generate", fake_revise_generate)

    task = Task(prompt="explain alpha", key_points=[])
    compiled = graph_module.build_graph()
    state = AgentState(**compiled.invoke(AgentState(task=task)))

    assert revise_called["n"] == 0  # revision never ran
    assert state.v2 is not None
    assert state.v2.text == state.v1.text
    assert state.v2.version.value == "v2"
    steps = [t.step for t in state.traces]
    assert "carry_forward" in steps
    assert "revise" not in steps


def test_graph_respects_max_iterations_zero(monkeypatch) -> None:
    """With max_iterations=0 the loop never revises, even when v1 fails."""
    _patch_models(
        monkeypatch,
        context="alpha beta gamma delta",
        v1_text="completely unrelated words here padding the length enough",
        v2_text="should not be produced",
    )
    task = Task(prompt="explain alpha", key_points=["alpha", "beta"])

    compiled = graph_module.build_graph()
    state = AgentState(**compiled.invoke(AgentState(task=task, max_iterations=0)))

    steps = [t.step for t in state.traces]
    assert "revise" not in steps
    assert "carry_forward" in steps
    assert state.v2 is not None and state.v2.text == state.v1.text


def test_graph_strict_threshold_forces_revision(monkeypatch) -> None:
    """A v1 that passes at a relaxed bar fails at a strict pass_threshold → revises."""
    _patch_models(
        monkeypatch,
        context="alpha beta gamma delta epsilon",
        v1_text="alpha beta gamma answer text padded to a decent length here",
        v2_text="alpha beta gamma delta epsilon fully grounded improved answer",
    )
    task = Task(prompt="explain alpha", key_points=["alpha", "beta"])
    compiled = graph_module.build_graph()

    relaxed = AgentState(**compiled.invoke(AgentState(task=task, pass_threshold=0.5)))
    strict = AgentState(**compiled.invoke(AgentState(task=task, pass_threshold=0.99)))

    relaxed_steps = [t.step for t in relaxed.traces]
    strict_steps = [t.step for t in strict.traces]
    assert "carry_forward" in relaxed_steps and "revise" not in relaxed_steps
    assert "revise" in strict_steps and "carry_forward" not in strict_steps
    assert strict.v1_eval is not None and not strict.v1_eval.passed


def test_generate_node_passes_provider_to_model(monkeypatch):
    """generate_node forwards AgentState.provider to the model layer."""
    from agents import graph as graph_mod
    from agents.state import AgentState
    from storage.models import Task

    captured = {}

    class _Completion:
        text = "answer"
        provider = "haiku"

    def _fake_generate(prompt, *, system=None, provider=None, **kwargs):
        captured["provider"] = provider
        return _Completion()

    monkeypatch.setattr(graph_mod, "generate", _fake_generate)
    state = AgentState(task=Task(prompt="q", source="user"), context="ctx", provider="haiku")
    graph_mod.generate_node(state)
    assert captured["provider"] == "haiku"
