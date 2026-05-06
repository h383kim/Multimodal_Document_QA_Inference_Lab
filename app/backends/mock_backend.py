"""Deterministic mock backend used for tests and CI.

Returns canned JSON answers from an in-memory answer book. Simulates plausible latency
so the benchmark layer has real numbers to crunch.
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from PIL.Image import Image

from app.backends.base import BackendResponse, ModelBackend


def _image_fingerprint(image: Image) -> str:
    """Stable hash over image bytes — used as the answer-book key."""
    buf = image.tobytes()
    return hashlib.sha1(buf).hexdigest()[:16]


GENERIC_FALLBACK = json.dumps(
    {
        "invoice_number": "INV-0000",
        "vendor_name": "Mock Vendor Inc.",
        "due_date": "2026-01-01",
        "total_amount": "$0.00",
    }
)


class MockBackend(ModelBackend):
    def __init__(
        self,
        answer_book: dict[str, str] | None = None,
        simulated_latency_ms: float = 25.0,
        simulated_tokens: int = 32,
        force_invalid_first_call: bool = False,
    ) -> None:
        """
        answer_book: maps image-fingerprint → raw model output (str).
        force_invalid_first_call: when true, the first generate() returns malformed
            JSON so retry behaviour can be exercised in tests.
        """
        self.answer_book = answer_book or {}
        self.simulated_latency_ms = simulated_latency_ms
        self.simulated_tokens = simulated_tokens
        self.force_invalid_first_call = force_invalid_first_call
        self._call_count = 0

    def generate(
        self,
        image: Image,
        prompt: str,
        generation_config: dict[str, Any] | None = None,
    ) -> BackendResponse:
        self._call_count += 1
        time.sleep(self.simulated_latency_ms / 1000.0)

        if self.force_invalid_first_call and self._call_count == 1:
            answer = "sorry, I can not produce JSON right now."
        else:
            key = _image_fingerprint(image)
            answer = self.answer_book.get(key, GENERIC_FALLBACK)

        ttft = self.simulated_latency_ms * 0.25
        total = self.simulated_latency_ms
        tokens = self.simulated_tokens
        tps = tokens / (total / 1000.0) if total > 0 else 0.0

        return BackendResponse(
            answer=answer,
            tokens_generated=tokens,
            time_to_first_token_ms=ttft,
            total_latency_ms=total,
            tokens_per_second=tps,
            peak_memory_mb=0.0,
            backend="mock",
            model="mock-vlm",
            quantization="none",
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "backend": "mock",
            "model": "mock-vlm",
            "quantization": "none",
            "device": "cpu",
        }

    def register_answer(self, image: Image, answer: str) -> None:
        """Helper for tests to attach an expected answer to an image."""
        self.answer_book[_image_fingerprint(image)] = answer
