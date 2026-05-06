"""FastAPI dependency providers (singletons that survive the app lifetime)."""
from __future__ import annotations

from functools import lru_cache

from app.backends.base import ModelBackend
from app.backends.mock_backend import MockBackend
from app.config import Settings, get_settings
from app.ingestion.document_store import DocumentStore


@lru_cache(maxsize=1)
def get_document_store() -> DocumentStore:
    settings = get_settings()
    return DocumentStore(max_image_edge=settings.max_image_edge)


@lru_cache(maxsize=8)
def _build_backend(backend_name: str, model: str | None, quantization: str | None) -> ModelBackend:
    if backend_name == "mock":
        return MockBackend()
    if backend_name == "transformers":
        # Lazy import: torch/transformers only required when actually requested.
        from app.backends.transformers_backend import TransformersBackend

        return TransformersBackend(
            model_id=model or get_settings().transformers_model_id,
            quantization=quantization or "fp16",
        )
    raise ValueError(f"unknown backend '{backend_name}'")


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
