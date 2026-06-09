"""Tests for the LLM judge (evaluators/judge.py).

The judge calls Haiku, so every test monkeypatches ``generate`` to return canned
text — no API calls, fully offline (mirrors tests/test_graph.py). We test the
parsing, the malformed-output fallback, and that the judge is forced onto the
Haiku provider with the rubric cached.
"""

from __future__ import annotations

from agents.llm import Completion, LLMError
from evaluators import judge as judge_module


def _fake_completion(text: str) -> Completion:
    return Completion(text=text, provider="haiku", model="test", latency_ms=1.0)


def _patch_generate(monkeypatch, text: str, captured: dict | None = None) -> None:
    def fake_generate(prompt, **kwargs):
        if captured is not None:
            captured.update(kwargs)
        return _fake_completion(text)

    monkeypatch.setattr(judge_module, "generate", fake_generate)


def test_judge_parses_valid_json(monkeypatch) -> None:
    _patch_generate(
        monkeypatch,
        '{"correctness": {"score": 0.8, "justification": "accurate"},'
        ' "completeness": {"score": 0.6, "justification": "partial"},'
        ' "clarity": {"score": 0.9, "justification": "clear"}}',
    )
    scores = judge_module.judge_response("task", "answer", "context")

    by_name = {s.name: s for s in scores}
    assert set(by_name) == {"correctness", "completeness", "clarity"}
    assert by_name["correctness"].score == 0.8
    assert by_name["clarity"].justification == "clear"


def test_judge_extracts_json_from_surrounding_prose(monkeypatch) -> None:
    """Real models sometimes wrap JSON in prose or code fences; we recover it."""
    _patch_generate(
        monkeypatch,
        'Here is my evaluation:\n```json\n'
        '{"correctness": {"score": 1.0, "justification": "x"},'
        ' "completeness": {"score": 1.0, "justification": "x"},'
        ' "clarity": {"score": 1.0, "justification": "x"}}\n```\nThanks!',
    )
    scores = judge_module.judge_response("task", "answer")
    assert len(scores) == 3


def test_judge_clamps_out_of_range_scores(monkeypatch) -> None:
    _patch_generate(
        monkeypatch,
        '{"correctness": {"score": 5, "justification": "x"},'
        ' "completeness": {"score": -2, "justification": "x"},'
        ' "clarity": {"score": 0.5, "justification": "x"}}',
    )
    by_name = {s.name: s for s in judge_module.judge_response("t", "a")}
    assert by_name["correctness"].score == 1.0
    assert by_name["completeness"].score == 0.0
    assert by_name["clarity"].score == 0.5


def test_judge_returns_empty_on_unparseable_output(monkeypatch) -> None:
    """Malformed output after retries degrades gracefully to no judge scores."""
    _patch_generate(monkeypatch, "I cannot produce JSON, sorry.")
    assert judge_module.judge_response("task", "answer") == []


def test_judge_returns_empty_when_provider_fails(monkeypatch) -> None:
    def boom(prompt, **kwargs):
        raise LLMError("no api key")

    monkeypatch.setattr(judge_module, "generate", boom)
    assert judge_module.judge_response("task", "answer") == []


def test_judge_forces_haiku_with_caching(monkeypatch) -> None:
    captured: dict = {}
    _patch_generate(
        monkeypatch,
        '{"correctness": {"score": 1, "justification": "x"},'
        ' "completeness": {"score": 1, "justification": "x"},'
        ' "clarity": {"score": 1, "justification": "x"}}',
        captured,
    )
    judge_module.judge_response("task", "answer")
    assert captured["provider"] == "haiku"
    assert captured["cache_system"] is True
