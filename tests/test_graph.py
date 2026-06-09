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
from storage.models import Task


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


def test_graph_runs_retrieve_then_generate(monkeypatch) -> None:
    """The graph retrieves, then generates a grounded v1, emitting both traces."""
    fake_chunks = [
        KbChunk(doc_id="llm-as-judge", title="What it is", text="llm as judge scores responses"),
    ]
    monkeypatch.setattr(graph_module, "load_kb", lambda _dir: fake_chunks)
    monkeypatch.setattr(
        graph_module,
        "generate",
        lambda prompt, system=None, **kw: Completion(
            text="A grounded answer.", provider="ollama", model="test", latency_ms=1.0
        ),
    )

    compiled = graph_module.build_graph()
    result = compiled.invoke(AgentState(task=Task(prompt="what is llm as judge")))
    state = AgentState(**result)

    assert state.v1 is not None
    assert state.v1.text == "A grounded answer."
    assert state.v1.retrieved_doc_ids == ["llm-as-judge"]
    assert [t.step for t in state.traces] == ["retrieve", "generate"]
