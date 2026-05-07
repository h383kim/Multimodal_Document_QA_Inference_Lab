"""Round-trip tests for the SQLite benchmark results store."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from app.benchmarking.results_store import append_run, get_rows, init_db, list_runs


def _fake_report(dataset: str = "data/sample") -> dict:
    return {
        "run_started_at": datetime(2026, 5, 7, 12, 0, tzinfo=timezone.utc).isoformat(),
        "dataset": dataset,
        "backend": {
            "backend": "mock",
            "model": "mock-vlm",
            "quantization": "fp16",
        },
        "aggregate": {
            "n": 2,
            "p50_latency_ms": 12.5,
            "p95_latency_ms": 30.0,
            "mean_latency_ms": 15.0,
            "mean_ttft_ms": 4.0,
            "mean_tokens_per_second": 250.0,
            "schema_valid_rate": 1.0,
            "field_accuracy_mean": 0.875,
            "retry_rate": 0.0,
        },
        "per_row": [
            {
                "id": "row1",
                "schema": "invoice_extraction",
                "schema_valid": True,
                "retry_count": 0,
                "field_accuracy": 1.0,
                "ttft_ms": 4.0,
                "total_latency_ms": 12.0,
                "tokens_per_second": 260.0,
                "peak_memory_mb": 200.0,
            },
            {
                "id": "row2",
                "schema": "invoice_extraction",
                "schema_valid": True,
                "retry_count": 0,
                "field_accuracy": 0.75,
                "ttft_ms": 4.5,
                "total_latency_ms": 18.0,
                "tokens_per_second": 240.0,
                "peak_memory_mb": 210.0,
            },
        ],
    }


def test_append_and_list_runs(tmp_path: Path):
    db = tmp_path / "runs.sqlite"
    run_id = append_run(_fake_report(), db)

    runs = list_runs(db)
    assert len(runs) == 1
    run = runs[0]
    assert run["run_id"] == run_id
    assert run["backend"] == "mock"
    assert run["model"] == "mock-vlm"
    assert run["n"] == 2
    assert run["p50_latency_ms"] == 12.5
    assert run["schema_valid_rate"] == 1.0


def test_get_rows_returns_per_item_records(tmp_path: Path):
    db = tmp_path / "runs.sqlite"
    run_id = append_run(_fake_report(), db)
    rows = get_rows(db, run_id)

    assert [r["item_id"] for r in rows] == ["row1", "row2"]
    assert all(r["schema_valid"] == 1 for r in rows)
    assert rows[0]["field_accuracy"] == 1.0
    assert rows[1]["field_accuracy"] == 0.75


def test_list_runs_orders_newest_first(tmp_path: Path):
    db = tmp_path / "runs.sqlite"
    older = _fake_report()
    older["run_started_at"] = "2026-05-01T00:00:00+00:00"
    newer = _fake_report()
    newer["run_started_at"] = "2026-05-07T00:00:00+00:00"

    append_run(older, db)
    append_run(newer, db)

    runs = list_runs(db)
    assert runs[0]["timestamp"].startswith("2026-05-07")
    assert runs[1]["timestamp"].startswith("2026-05-01")


def test_list_runs_on_missing_db_returns_empty(tmp_path: Path):
    assert list_runs(tmp_path / "absent.sqlite") == []


def test_init_db_is_idempotent(tmp_path: Path):
    db = tmp_path / "runs.sqlite"
    init_db(db)
    init_db(db)  # second call must not raise
    assert db.exists()
