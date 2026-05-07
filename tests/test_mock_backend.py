"""MockBackend contract tests."""

from __future__ import annotations

from app.backends.base import BackendResponse
from app.backends.mock_backend import MockBackend

REQUIRED_KEYS = set(BackendResponse.__annotations__.keys())


def test_mock_returns_full_response_shape(small_image):
    backend = MockBackend(simulated_latency_ms=2.0)
    resp = backend.generate(small_image, "extract")
    assert set(resp.keys()) == REQUIRED_KEYS
    assert resp["backend"] == "mock"
    assert resp["model"] == "mock-vlm"
    assert resp["total_latency_ms"] >= 0
    assert resp["tokens_generated"] > 0


def test_mock_is_deterministic(small_image):
    backend = MockBackend(simulated_latency_ms=2.0)
    backend.register_answer(small_image, '{"invoice_number": "INV-XYZ"}')
    a = backend.generate(small_image, "extract")
    b = backend.generate(small_image, "extract")
    assert a["answer"] == b["answer"] == '{"invoice_number": "INV-XYZ"}'


def test_mock_force_invalid_first_call(small_image):
    backend = MockBackend(simulated_latency_ms=2.0, force_invalid_first_call=True)
    bad = backend.generate(small_image, "extract")
    good = backend.generate(small_image, "extract")
    assert "json" not in bad["answer"].lower() or not bad["answer"].strip().startswith("{")
    assert good["answer"].strip().startswith("{")
