"""Tests for the revision prompt (agents/prompts.py) and revise step (feedback/improve.py).

The revise step is tested offline: the model call is monkeypatched so no Ollama
is needed. The key invariant is that revision is driven by *feedback + context*
only — the golden ``expected`` answer never reaches the prompt (no leakage), and
revision uses the default free provider, never the paid judge.
"""

from __future__ import annotations

from agents.llm import Completion
from agents.prompts import REVISION_PROMPT_VERSION, build_revision_prompt
from feedback import improve as improve_module
from feedback.improve import revise


def test_build_revision_prompt_includes_feedback_context_and_previous() -> None:
    prompt = build_revision_prompt(
        question="What is RAG?",
        context="[rag] retrieval grounding documents",
        previous_answer="RAG is a thing.",
        feedback="- Make sure to address: retriever, generator.",
    )
    assert "What is RAG?" in prompt
    assert "[rag] retrieval grounding documents" in prompt
    assert "RAG is a thing." in prompt
    assert "retriever, generator" in prompt
    assert REVISION_PROMPT_VERSION == "v1"


def test_revise_calls_model_with_feedback_and_no_expected_leakage(monkeypatch) -> None:
    """revise() builds the prompt from feedback + context, never the gold answer."""
    captured: dict[str, str] = {}

    def fake_generate(prompt, system=None, **kw):
        captured["prompt"] = prompt
        return Completion(
            text="A revised, grounded answer.", provider="ollama", model="test", latency_ms=1.0
        )

    monkeypatch.setattr(improve_module, "generate", fake_generate)

    text = revise(
        question="What is RAG?",
        context="[rag] retrieval grounding",
        previous_answer="RAG is a thing.",
        feedback="- Make sure to address: retriever.",
    )

    assert text == "A revised, grounded answer."
    assert "retriever" in captured["prompt"]
    assert "[rag] retrieval grounding" in captured["prompt"]
    # revise() has no parameter for the gold answer, so leakage is structurally impossible.


def test_revise_uses_default_provider_not_judge(monkeypatch) -> None:
    """Revision runs on the default (free) provider — never forced to Haiku."""
    seen: dict[str, object] = {}

    def fake_generate(prompt, system=None, provider=None, **kw):
        seen["provider"] = provider
        return Completion(text="ok", provider="ollama", model="test", latency_ms=1.0)

    monkeypatch.setattr(improve_module, "generate", fake_generate)
    revise(question="q", context="c", previous_answer="p", feedback="f")
    # revise must not force provider="haiku" (cost rule): it leaves it to settings.
    assert seen["provider"] in (None, "ollama")


def test_revise_forwards_explicit_provider(monkeypatch) -> None:
    """revise() forwards an explicit provider kwarg to the model layer unchanged."""
    seen: dict[str, object] = {}

    class _Completion:
        text = "revised"
        provider = "haiku"

    def _fake_generate(prompt, *, system=None, provider=None, settings=None, **kwargs):
        seen["provider"] = provider
        return _Completion()

    monkeypatch.setattr("feedback.improve.generate", _fake_generate)
    revise(question="q", context="c", previous_answer="a", feedback="f", provider="haiku")
    assert seen["provider"] == "haiku"
