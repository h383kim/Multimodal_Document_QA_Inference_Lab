"""Backend interface contract.

Every backend (mock, transformers, vllm, llama.cpp) returns the same dict shape so the
benchmarking layer can compare apples to apples. See CLAUDE.md for the contract spec.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict

from PIL.Image import Image


class BackendResponse(TypedDict):
    answer: str
    tokens_generated: int
    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_per_second: float
    peak_memory_mb: float
    backend: str
    model: str
    quantization: str


class ModelBackend(ABC):
    @abstractmethod
    def generate(
        self,
        image: Image,
        prompt: str,
        generation_config: dict[str, Any] | None = None,
    ) -> BackendResponse: ...

    @abstractmethod
    def get_model_info(self) -> dict[str, Any]: ...
