"""Benchmark runner over a fixture dataset."""

from __future__ import annotations

import json
from pathlib import Path

from app.backends.mock_backend import MockBackend
from app.benchmarking.runner import run_benchmark

REQUIRED_AGGREGATE_KEYS = {
    "n",
    "p50_latency_ms",
    "p95_latency_ms",
    "mean_latency_ms",
    "mean_ttft_ms",
    "mean_tokens_per_second",
    "schema_valid_rate",
    "field_accuracy_mean",
    "retry_rate",
}


def test_runner_aggregates_metrics(fixture_dataset, answer_book_for_fixture, tmp_path):
    backend = MockBackend(simulated_latency_ms=2.0, answer_book=answer_book_for_fixture)
    output = tmp_path / "report.json"

    report = run_benchmark(fixture_dataset, backend, output_path=output)

    assert REQUIRED_AGGREGATE_KEYS.issubset(report["aggregate"].keys())
    assert report["aggregate"]["n"] == 2
    assert report["aggregate"]["schema_valid_rate"] == 1.0
    assert report["aggregate"]["field_accuracy_mean"] == 1.0
    assert report["aggregate"]["retry_rate"] == 0.0

    assert output.exists()
    on_disk = json.loads(Path(output).read_text())
    assert on_disk["aggregate"]["n"] == 2


def test_runner_handles_invalid_answers(fixture_dataset, tmp_path):
    backend = MockBackend(
        simulated_latency_ms=1.0
    )  # no answer book → falls back to wrong-but-valid JSON
    output = tmp_path / "report.json"

    report = run_benchmark(fixture_dataset, backend, output_path=output, max_retries=0)
    assert report["aggregate"]["schema_valid_rate"] == 1.0  # still valid JSON
    assert report["aggregate"]["field_accuracy_mean"] < 1.0  # but wrong content
