"""Tests for the dataset loader (datasets/loader.py).

The loader reads a frozen JSONL dataset into validated ``Task`` objects. It is a
pure function over a file, so it is tested against tiny temp fixtures — no model
calls, no real dataset files.
"""

from __future__ import annotations

from pathlib import Path

from datasets.generate import build_synthetic
from datasets.loader import load_dataset
from storage.models import Task


def _write_jsonl(path: Path, *lines: str) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_dataset_reads_records_as_tasks(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    _write_jsonl(
        path,
        '{"prompt": "what is RAG?", "source": "golden", "expected": "grounding",'
        ' "key_points": ["retriever", "generator"]}',
        '{"prompt": "what is a judge?", "source": "golden"}',
    )

    tasks = load_dataset(path)

    assert len(tasks) == 2
    assert all(isinstance(t, Task) for t in tasks)
    assert tasks[0].prompt == "what is RAG?"
    assert tasks[0].source == "golden"
    assert tasks[0].key_points == ["retriever", "generator"]
    # Missing optional fields fall back to Task defaults.
    assert tasks[1].expected is None
    assert tasks[1].key_points == []


def test_load_dataset_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "d.jsonl"
    path.write_text(
        '{"prompt": "a"}\n\n   \n{"prompt": "b"}\n', encoding="utf-8"
    )

    tasks = load_dataset(path)

    assert [t.prompt for t in tasks] == ["a", "b"]


def test_load_dataset_filters_by_split(tmp_path: Path) -> None:
    """A ``split`` key partitions golden into dev vs held-out test."""
    path = tmp_path / "d.jsonl"
    _write_jsonl(
        path,
        '{"prompt": "dev one", "split": "dev"}',
        '{"prompt": "test one", "split": "test"}',
        '{"prompt": "dev two", "split": "dev"}',
    )

    dev = load_dataset(path, split="dev")
    test = load_dataset(path, split="test")

    assert [t.prompt for t in dev] == ["dev one", "dev two"]
    assert [t.prompt for t in test] == ["test one"]
    # ``split`` is a dataset-only key; it must not leak onto the Task model.
    assert not hasattr(dev[0], "split")


def test_build_synthetic_is_deterministic_and_frozen_shape() -> None:
    """The generator is pure: same inputs -> identical, label-free synthetic tasks."""
    first = build_synthetic(limit=40)
    second = build_synthetic(limit=40)

    assert len(first) == 40
    # Deterministic: prompts are identical and in the same order across calls.
    assert [t.prompt for t in first] == [t.prompt for t in second]
    # Prompts are distinct (no accidental duplicates from the template grid).
    assert len({t.prompt for t in first}) == 40
    # Synthetic inputs are unlabeled: source tagged, no expected/key_points.
    assert all(t.source == "synthetic" for t in first)
    assert all(t.expected is None and t.key_points == [] for t in first)
