"""FastAPI dependency providers (singletons that survive the app lifetime)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import structlog
from PIL import Image

from app.backends.base import ModelBackend
from app.backends.mock_backend import MockBackend
from app.config import Settings, get_settings
from app.ingestion.document_store import DocumentStore

log = structlog.get_logger("app.api.deps")


@lru_cache(maxsize=1)
def get_document_store() -> DocumentStore:
    settings = get_settings()
    return DocumentStore(max_image_edge=settings.max_image_edge)


@lru_cache(maxsize=8)
def _build_backend(backend_name: str, model: str | None, quantization: str | None) -> ModelBackend:
    log.info(
        "backend.init",
        backend=backend_name,
        model=model,
        quantization=quantization,
    )
    if backend_name == "mock":
        return _build_mock_backend()
    if backend_name == "transformers":
        # Lazy import: torch/transformers only required when actually requested.
        from app.backends.transformers_backend import TransformersBackend

        return TransformersBackend(
            model_id=model or get_settings().transformers_model_id,
            quantization=quantization or "fp16",
        )
    raise ValueError(f"unknown backend '{backend_name}'")


def _build_mock_backend() -> MockBackend:
    backend = MockBackend()
    _register_sample_invoice_answers(backend)
    return backend


def _register_sample_invoice_answers(
    backend: MockBackend,
    dataset_dir: Path | None = None,
) -> int:
    """Load generated sample invoice labels into the mock backend when present."""
    dataset_dir = dataset_dir or get_settings().sample_invoice_dir
    labels_path = dataset_dir / "labels.jsonl"
    if not labels_path.exists():
        return 0

    registered = 0
    for line in labels_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        image_path = dataset_dir / row["file"]
        expected = row.get("expected")
        if not image_path.exists() or expected is None:
            continue
        with Image.open(image_path) as image:
            backend.register_answer(image.convert("RGB"), json.dumps(expected))
        registered += 1
    return registered


def get_backend(
    backend_name: str | None = None,
    model: str | None = None,
    quantization: str | None = None,
) -> ModelBackend:
    settings = get_settings()
    return _build_backend(
        backend_name or settings.default_backend,
        model,
        quantization,
    )


def get_app_settings() -> Settings:
    return get_settings()
