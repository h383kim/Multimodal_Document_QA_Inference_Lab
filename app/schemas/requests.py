"""API request/response models."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


OutputMode = Literal["natural_language", "json", "field_extraction"]


class QARequest(BaseModel):
    file_id: str
    question: str
    output_mode: OutputMode = "json"
    schema_name: str | None = "invoice_extraction"
    backend: str | None = None
    model: str | None = None
    quantization: str | None = None
    max_retries: int = 2


class QAMetrics(BaseModel):
    ttft_ms: float
    total_latency_ms: float
    tokens_per_second: float
    peak_memory_mb: float
    retry_count: int
    schema_valid: bool


class QAResponse(BaseModel):
    answer: dict[str, Any] | str
    metrics: QAMetrics
    backend: str
    model: str
    quantization: str


class BenchmarkRunRequest(BaseModel):
    dataset: str
    backend: str = "mock"
    model: str | None = None
    quantization: str = "fp16"
    output: str | None = None
    max_retries: int = 2


class UploadResponse(BaseModel):
    file_id: str
    num_pages: int
    document_type: str
