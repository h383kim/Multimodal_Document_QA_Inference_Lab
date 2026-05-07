"""SQLite-backed persistence for benchmark runs.

The runner remains pure (returns a dict). Callers — the CLI, the API endpoint,
and the dashboard — opt into persistence via ``append_run`` / ``list_runs``.

Schema (no migrations; ``CREATE TABLE IF NOT EXISTS`` only):

    runs(
        run_id TEXT PRIMARY KEY,        -- uuid4 hex
        timestamp TEXT NOT NULL,        -- ISO 8601 UTC
        dataset TEXT NOT NULL,
        backend TEXT NOT NULL,
        model TEXT,
        quantization TEXT,
        n INTEGER NOT NULL,
        p50_latency_ms REAL,
        p95_latency_ms REAL,
        mean_latency_ms REAL,
        mean_ttft_ms REAL,
        mean_tokens_per_second REAL,
        schema_valid_rate REAL,
        field_accuracy_mean REAL,
        retry_rate REAL
    )

    rows(
        run_id TEXT NOT NULL,
        item_id TEXT NOT NULL,
        schema TEXT,
        schema_valid INTEGER,
        retry_count INTEGER,
        field_accuracy REAL,
        ttft_ms REAL,
        total_latency_ms REAL,
        tokens_per_second REAL,
        peak_memory_mb REAL,
        FOREIGN KEY(run_id) REFERENCES runs(run_id)
    )
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_RUNS_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    dataset TEXT NOT NULL,
    backend TEXT NOT NULL,
    model TEXT,
    quantization TEXT,
    n INTEGER NOT NULL,
    p50_latency_ms REAL,
    p95_latency_ms REAL,
    mean_latency_ms REAL,
    mean_ttft_ms REAL,
    mean_tokens_per_second REAL,
    schema_valid_rate REAL,
    field_accuracy_mean REAL,
    retry_rate REAL
)
"""

_ROWS_DDL = """
CREATE TABLE IF NOT EXISTS rows (
    run_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    schema TEXT,
    schema_valid INTEGER,
    retry_count INTEGER,
    field_accuracy REAL,
    ttft_ms REAL,
    total_latency_ms REAL,
    tokens_per_second REAL,
    peak_memory_mb REAL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id)
)
"""

_RUN_COLUMNS = (
    "run_id",
    "timestamp",
    "dataset",
    "backend",
    "model",
    "quantization",
    "n",
    "p50_latency_ms",
    "p95_latency_ms",
    "mean_latency_ms",
    "mean_ttft_ms",
    "mean_tokens_per_second",
    "schema_valid_rate",
    "field_accuracy_mean",
    "retry_rate",
)

_ROW_COLUMNS = (
    "run_id",
    "item_id",
    "schema",
    "schema_valid",
    "retry_count",
    "field_accuracy",
    "ttft_ms",
    "total_latency_ms",
    "tokens_per_second",
    "peak_memory_mb",
)


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Path) -> None:
    """Create schema if missing. Idempotent."""
    with _connect(db_path) as conn:
        conn.execute(_RUNS_DDL)
        conn.execute(_ROWS_DDL)
        conn.commit()


def append_run(report: dict[str, Any], db_path: Path) -> str:
    """Persist a benchmark report. Returns the assigned run_id."""
    init_db(db_path)
    run_id = uuid.uuid4().hex
    aggregate = report.get("aggregate", {})
    backend_info = report.get("backend", {})
    timestamp = report.get("run_started_at") or datetime.now(timezone.utc).isoformat()

    run_record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "dataset": str(report.get("dataset", "")),
        "backend": backend_info.get("backend", "unknown"),
        "model": backend_info.get("model"),
        "quantization": backend_info.get("quantization"),
        "n": aggregate.get("n", 0),
        "p50_latency_ms": aggregate.get("p50_latency_ms"),
        "p95_latency_ms": aggregate.get("p95_latency_ms"),
        "mean_latency_ms": aggregate.get("mean_latency_ms"),
        "mean_ttft_ms": aggregate.get("mean_ttft_ms"),
        "mean_tokens_per_second": aggregate.get("mean_tokens_per_second"),
        "schema_valid_rate": aggregate.get("schema_valid_rate"),
        "field_accuracy_mean": aggregate.get("field_accuracy_mean"),
        "retry_rate": aggregate.get("retry_rate"),
    }

    row_records = [
        {
            "run_id": run_id,
            "item_id": row.get("id", ""),
            "schema": row.get("schema"),
            "schema_valid": int(bool(row.get("schema_valid"))),
            "retry_count": row.get("retry_count", 0),
            "field_accuracy": row.get("field_accuracy"),
            "ttft_ms": row.get("ttft_ms"),
            "total_latency_ms": row.get("total_latency_ms"),
            "tokens_per_second": row.get("tokens_per_second"),
            "peak_memory_mb": row.get("peak_memory_mb"),
        }
        for row in report.get("per_row", [])
    ]

    with _connect(db_path) as conn:
        placeholders = ",".join(":" + col for col in _RUN_COLUMNS)
        conn.execute(
            f"INSERT INTO runs ({','.join(_RUN_COLUMNS)}) VALUES ({placeholders})",
            run_record,
        )
        if row_records:
            row_placeholders = ",".join(":" + col for col in _ROW_COLUMNS)
            conn.executemany(
                f"INSERT INTO rows ({','.join(_ROW_COLUMNS)}) VALUES ({row_placeholders})",
                row_records,
            )
        conn.commit()

    return run_id


def list_runs(db_path: Path) -> list[dict[str, Any]]:
    """Return all runs ordered newest-first. Empty list if the DB doesn't exist."""
    if not db_path.exists():
        return []
    init_db(db_path)
    with _connect(db_path) as conn:
        cursor = conn.execute("SELECT * FROM runs ORDER BY timestamp DESC")
        return [dict(row) for row in cursor.fetchall()]


def get_rows(db_path: Path, run_id: str) -> list[dict[str, Any]]:
    """Return per-item rows for a given run, in insertion order."""
    if not db_path.exists():
        return []
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "SELECT * FROM rows WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
