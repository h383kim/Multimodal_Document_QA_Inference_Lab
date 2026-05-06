"""POST /benchmarks/run — synchronous benchmark trigger.

The MVP runs synchronously and returns the aggregate result. Async/queued runs are a
stretch goal listed in the design doc.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.api.deps import get_backend
from app.benchmarking.runner import run_benchmark
from app.schemas.requests import BenchmarkRunRequest


router = APIRouter(prefix="/benchmarks", tags=["benchmarks"])


@router.post("/run")
def run(request: BenchmarkRunRequest) -> dict[str, Any]:
    backend = get_backend(request.backend, request.model, request.quantization)
    return run_benchmark(
        dataset_dir=request.dataset,
        backend=backend,
        output_path=request.output,
        max_retries=request.max_retries,
    )
