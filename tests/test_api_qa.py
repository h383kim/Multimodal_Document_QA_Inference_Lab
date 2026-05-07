"""End-to-end API smoke via FastAPI TestClient."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.deps import _build_backend, get_document_store
from app.backends.mock_backend import MockBackend
from app.main import create_app


def _client_with_canned_answer(small_image):
    """Build a client + register a known answer for the test image."""
    app = create_app()
    client = TestClient(app)

    # Reach into the lru_cache to install a backend with the right answer book.
    backend = MockBackend(simulated_latency_ms=1.0)
    backend.register_answer(
        small_image,
        '{"invoice_number": "INV-API-1", "vendor_name": "Test", "due_date": "2026-01-01", "total_amount": "$5.00"}',
    )
    _build_backend.cache_clear()
    get_document_store.cache_clear()
    # Prime the cache so the backend factory returns our instance.
    from functools import lru_cache

    import app.api.deps as deps

    deps._build_backend = lru_cache(maxsize=1)(lambda *_a, **_k: backend)  # type: ignore[assignment]
    return client, backend


def test_health(small_image):
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_upload_then_qa_roundtrip(small_image, small_image_bytes):
    client, _ = _client_with_canned_answer(small_image)
    r = client.post(
        "/documents/upload",
        files={"file": ("test.png", small_image_bytes, "image/png")},
    )
    assert r.status_code == 200, r.text
    file_id = r.json()["file_id"]
    assert r.json()["num_pages"] == 1

    r2 = client.post(
        "/qa",
        json={
            "file_id": file_id,
            "question": "Extract fields",
            "output_mode": "json",
            "schema_name": "invoice_extraction",
            "backend": "mock",
        },
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["answer"]["invoice_number"] == "INV-API-1"
    assert body["metrics"]["schema_valid"] is True
    assert body["backend"] == "mock"


def test_qa_unknown_file_id(small_image):
    client, _ = _client_with_canned_answer(small_image)
    r = client.post(
        "/qa",
        json={
            "file_id": "does-not-exist",
            "question": "?",
            "output_mode": "json",
            "schema_name": "invoice_extraction",
            "backend": "mock",
        },
    )
    assert r.status_code == 404
