"""POST /benchmarks/run — synchronous benchmark trigger.

The MVP runs synchronously and returns the aggregate result. Async/queued runs are a
stretch goal listed in the design doc.
"""

from __future__ import annotations

import time
from typing import Any

import structlog
from fastapi import APIRouter

from app.api.deps import get_backend
from app.benchmarking.runner import run_benchmark
from app.schemas.requests import BenchmarkRunRequest

router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])
log = structlog.get_logger("app.api.benchmark")


@router.post("/run")
def run(request: BenchmarkRunRequest) -> dict[str, Any]:
    backend = get_backend(request.backend, request.model, request.quantization)
    log.info(
        "benchmark.started",
        dataset=str(request.dataset),
        backend=request.backend,
        model=request.model,
        quantization=request.quantization,
    )
    start = time.perf_counter()
    report = run_benchmark(
        dataset_dir=request.dataset,
        backend=backend,
        output_path=request.output,
        max_retries=request.max_retries,
    )
    aggregate = report.get("aggregate", {})
    log.info(
        "benchmark.completed",
        duration_s=round(time.perf_counter() - start, 2),
        items=aggregate.get("n"),
        schema_valid_rate=aggregate.get("schema_valid_rate"),
        field_accuracy_mean=aggregate.get("field_accuracy_mean"),
    )
    return report
