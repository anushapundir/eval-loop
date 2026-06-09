"""Load frozen JSONL datasets into validated ``Task`` objects.

The golden and synthetic datasets are committed JSONL files — one JSON record
per line, fields mapping directly onto :class:`storage.models.Task` (``prompt``,
``source``, ``expected``, ``key_points``). Records may carry an extra ``split``
key ("dev" | "test") used only to partition the golden set into a dev split and
a held-out test split the feedback step (Day 4) must never see; ``split`` is a
dataset-only concern and never reaches the ``Task`` model.

This module is pure (a file in, a list out) so evaluation is reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

from storage.models import Task

# Task fields a dataset record may set; anything else (e.g. ``split``) is
# dataset metadata and is not passed to the model.
_TASK_FIELDS = frozenset({"id", "prompt", "source", "expected", "key_points"})


def load_dataset(path: Path, *, split: str | None = None) -> list[Task]:
    """Read a JSONL dataset file into a list of ``Task`` objects.

    Args:
        path: Path to the ``.jsonl`` dataset file.
        split: If given, keep only records whose ``split`` equals this value
            (used to select the golden dev or held-out test partition).

    Returns:
        The tasks in file order, blank lines skipped.
    """
    tasks: list[Task] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if split is not None and record.get("split") != split:
            continue
        fields = {k: v for k, v in record.items() if k in _TASK_FIELDS}
        tasks.append(Task(**fields))
    return tasks
