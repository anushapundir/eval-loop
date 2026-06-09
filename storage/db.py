"""SQLite storage — the ONLY module that touches the database (CLAUDE.md §6).

Every other layer reads and writes through these functions, against the schema
defined by ``storage/models.py``. Complex fields (lists, dicts) are stored as
JSON text; timestamps as ISO-8601 strings. Tables are created idempotently.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from config.settings import get_settings
from storage.models import (
    AgentResponse,
    CriterionScore,
    EvalResult,
    Experiment,
    ResponseVersion,
    Task,
    Trace,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    prompt      TEXT NOT NULL,
    source      TEXT NOT NULL,
    expected    TEXT,
    key_points  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS responses (
    id                TEXT PRIMARY KEY,
    task_id           TEXT NOT NULL,
    version           TEXT NOT NULL,
    text              TEXT NOT NULL,
    retrieved_doc_ids TEXT NOT NULL,
    model_provider    TEXT NOT NULL,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS traces (
    id          TEXT PRIMARY KEY,
    task_id     TEXT NOT NULL,
    step        TEXT NOT NULL,
    response_id TEXT,
    provider    TEXT NOT NULL,
    latency_ms  REAL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
    id             TEXT PRIMARY KEY,
    task_id        TEXT NOT NULL,
    response_id    TEXT NOT NULL,
    version        TEXT NOT NULL,
    deterministic  TEXT NOT NULL,
    judge          TEXT NOT NULL,
    overall_score  REAL NOT NULL,
    passed         INTEGER NOT NULL,
    judged         INTEGER NOT NULL,
    experiment_id  TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiments (
    id                TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    prompt_version    TEXT NOT NULL,
    model_provider    TEXT NOT NULL,
    n_tasks           INTEGER NOT NULL,
    n_judged          INTEGER NOT NULL,
    mean_v1           REAL,
    mean_v2           REAL,
    improvement_delta REAL,
    notes             TEXT,
    created_at        TEXT NOT NULL
);
"""


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Yield a SQLite connection with row access by column name."""
    settings = get_settings()
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    """Create all tables if they do not exist."""
    with connect(db_path) as conn:
        conn.executescript(_SCHEMA)


# --- writes ----------------------------------------------------------------


def write_task(task: Task, db_path: Path | None = None) -> None:
    """Insert or replace a task record."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO tasks VALUES (?,?,?,?,?,?)",
            (
                task.id,
                task.prompt,
                task.source,
                task.expected,
                json.dumps(task.key_points),
                task.created_at.isoformat(),
            ),
        )


def write_response(response: AgentResponse, db_path: Path | None = None) -> None:
    """Insert or replace an agent response record."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO responses VALUES (?,?,?,?,?,?,?)",
            (
                response.id,
                response.task_id,
                response.version.value,
                response.text,
                json.dumps(response.retrieved_doc_ids),
                response.model_provider,
                response.created_at.isoformat(),
            ),
        )


def write_trace(trace: Trace, db_path: Path | None = None) -> None:
    """Insert or replace a trace record."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?)",
            (
                trace.id,
                trace.task_id,
                trace.step,
                trace.response_id,
                trace.provider,
                trace.latency_ms,
                json.dumps(trace.payload),
                trace.created_at.isoformat(),
            ),
        )


def write_eval_result(result: EvalResult, db_path: Path | None = None) -> None:
    """Insert or replace an evaluation result record."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO eval_results VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                result.id,
                result.task_id,
                result.response_id,
                result.version.value,
                json.dumps([s.model_dump() for s in result.deterministic]),
                json.dumps([s.model_dump() for s in result.judge]),
                result.overall_score,
                int(result.passed),
                int(result.judged),
                result.experiment_id,
                result.created_at.isoformat(),
            ),
        )


def write_experiment(experiment: Experiment, db_path: Path | None = None) -> None:
    """Insert or replace an experiment record."""
    with connect(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO experiments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                experiment.id,
                experiment.name,
                experiment.prompt_version,
                experiment.model_provider,
                experiment.n_tasks,
                experiment.n_judged,
                experiment.mean_v1,
                experiment.mean_v2,
                experiment.improvement_delta,
                experiment.notes,
                experiment.created_at.isoformat(),
            ),
        )


# --- reads -----------------------------------------------------------------


def get_trace(trace_id: str, db_path: Path | None = None) -> Trace | None:
    """Fetch a single trace by id, or None."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM traces WHERE id = ?", (trace_id,)
        ).fetchone()
    return _row_to_trace(row) if row else None


def list_traces(task_id: str | None = None, db_path: Path | None = None) -> list[Trace]:
    """List traces, optionally filtered by task id."""
    with connect(db_path) as conn:
        if task_id:
            rows = conn.execute(
                "SELECT * FROM traces WHERE task_id = ? ORDER BY created_at",
                (task_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM traces ORDER BY created_at"
            ).fetchall()
    return [_row_to_trace(r) for r in rows]


def get_response(response_id: str, db_path: Path | None = None) -> AgentResponse | None:
    """Fetch a single response by id, or None."""
    with connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM responses WHERE id = ?", (response_id,)
        ).fetchone()
    return _row_to_response(row) if row else None


def list_eval_results(
    experiment_id: str | None = None, db_path: Path | None = None
) -> list[EvalResult]:
    """List eval results, optionally filtered by experiment id."""
    with connect(db_path) as conn:
        if experiment_id:
            rows = conn.execute(
                "SELECT * FROM eval_results WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM eval_results").fetchall()
    return [_row_to_eval_result(r) for r in rows]


def list_experiments(db_path: Path | None = None) -> list[Experiment]:
    """List all experiments, newest first."""
    with connect(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_experiment(r) for r in rows]


# --- row -> model helpers --------------------------------------------------


def _row_to_trace(row: sqlite3.Row) -> Trace:
    return Trace(
        id=row["id"],
        task_id=row["task_id"],
        step=row["step"],
        response_id=row["response_id"],
        provider=row["provider"],
        latency_ms=row["latency_ms"],
        payload=json.loads(row["payload"]),
        created_at=row["created_at"],
    )


def _row_to_response(row: sqlite3.Row) -> AgentResponse:
    return AgentResponse(
        id=row["id"],
        task_id=row["task_id"],
        version=ResponseVersion(row["version"]),
        text=row["text"],
        retrieved_doc_ids=json.loads(row["retrieved_doc_ids"]),
        model_provider=row["model_provider"],
        created_at=row["created_at"],
    )


def _row_to_eval_result(row: sqlite3.Row) -> EvalResult:
    return EvalResult(
        id=row["id"],
        task_id=row["task_id"],
        response_id=row["response_id"],
        version=ResponseVersion(row["version"]),
        deterministic=[CriterionScore(**s) for s in json.loads(row["deterministic"])],
        judge=[CriterionScore(**s) for s in json.loads(row["judge"])],
        overall_score=row["overall_score"],
        passed=bool(row["passed"]),
        judged=bool(row["judged"]),
        experiment_id=row["experiment_id"],
        created_at=row["created_at"],
    )


def _row_to_experiment(row: sqlite3.Row) -> Experiment:
    return Experiment(
        id=row["id"],
        name=row["name"],
        prompt_version=row["prompt_version"],
        model_provider=row["model_provider"],
        n_tasks=row["n_tasks"],
        n_judged=row["n_judged"],
        mean_v1=row["mean_v1"],
        mean_v2=row["mean_v2"],
        improvement_delta=row["improvement_delta"],
        notes=row["notes"],
        created_at=row["created_at"],
    )
