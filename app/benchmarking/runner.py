"""Benchmark runner — iterates a labeled dataset and produces an aggregate report.

A "dataset" is a directory shaped:
  data/<name>/
    images/...png
    labels.jsonl   # one row per example: {id, file, question, expected, schema}
"""

from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from app.backends.base import ModelBackend
from app.evals.field_match import per_field_accuracy, record_accuracy
from app.evals.metrics import fraction, mean, percentile
from app.evals.schema_validation import parse_json_with_retry
from app.schemas.outputs import get_schema


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _build_prompt(question: str, schema_name: str) -> str:
    schema_cls = get_schema(schema_name)
    keys = list(schema_cls.model_fields.keys())
    return (
        f"You are a document understanding assistant. Question: {question}\n\n"
        f"Reply with a single JSON object containing exactly these keys: {keys}. "
        "If a value is unknown, use null. Do not include prose."
    )


def run_benchmark(
    dataset_dir: str | Path,
    backend: ModelBackend,
    output_path: str | Path | None = None,
    max_retries: int = 2,
) -> dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    labels_path = dataset_dir / "labels.jsonl"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.jsonl not found at {labels_path}")

    info = backend.get_model_info()
    per_row: list[dict[str, Any]] = []

    for row in _iter_jsonl(labels_path):
        image_path = (dataset_dir / row["file"]).resolve()
        with image_path.open("rb") as f:
            image = Image.open(io.BytesIO(f.read())).convert("RGB")

        prompt = _build_prompt(row["question"], row["schema"])
        schema_cls = get_schema(row["schema"])

        result = parse_json_with_retry(
            backend=backend,
            image=image,
            prompt=prompt,
            schema=schema_cls,
            max_retries=max_retries,
        )

        backend_resp = result.last_backend_response
        predicted = result.parsed.model_dump() if result.parsed else None
        expected = row.get("expected", {})
        acc = record_accuracy(predicted, expected)
        field_breakdown = (
            per_field_accuracy(predicted, expected)
            if predicted is not None
            else {k: False for k in expected}
        )

        per_row.append(
            {
                "id": row["id"],
                "schema": row["schema"],
                "predicted": predicted,
                "raw_answer": result.raw_answer,
                "expected": expected,
                "schema_valid": result.schema_valid,
                "retry_count": result.retry_count,
                "field_accuracy": acc,
                "fields": field_breakdown,
                "ttft_ms": backend_resp["time_to_first_token_ms"],
                "total_latency_ms": backend_resp["total_latency_ms"],
                "tokens_per_second": backend_resp["tokens_per_second"],
                "peak_memory_mb": backend_resp["peak_memory_mb"],
            }
        )

    latencies = [r["total_latency_ms"] for r in per_row]
    ttfts = [r["ttft_ms"] for r in per_row]
    tps_values = [r["tokens_per_second"] for r in per_row]

    aggregate = {
        "n": len(per_row),
        "p50_latency_ms": percentile(latencies, 50),
        "p95_latency_ms": percentile(latencies, 95),
        "mean_latency_ms": mean(latencies),
        "mean_ttft_ms": mean(ttfts),
        "mean_tokens_per_second": mean(tps_values),
        "schema_valid_rate": fraction(r["schema_valid"] for r in per_row),
        "field_accuracy_mean": mean(r["field_accuracy"] for r in per_row),
        "retry_rate": fraction(r["retry_count"] > 0 for r in per_row),
    }

    report = {
        "run_started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_dir),
        "backend": info,
        "aggregate": aggregate,
        "per_row": per_row,
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2))

    return report
