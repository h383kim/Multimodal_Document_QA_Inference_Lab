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


def test_qa_with_routing_attaches_route_decision(
    small_image, small_image_bytes, tmp_path, monkeypatch
):
    """Routing on /qa should attach a route_decision and append a JSONL log entry."""
    monkeypatch.setenv("MMI_RESULTS_DIR", str(tmp_path))

    # The Router runs OCR by default; force it off so this test doesn't need pytesseract.
    from app.api import routes_qa
    from app.routing.router import Router

    routes_qa._get_router.cache_clear()
    monkeypatch.setattr(routes_qa, "_get_router", lambda: Router(run_ocr_on_route=False))

    # Clear the settings lru_cache so MMI_RESULTS_DIR takes effect.
    from app.config import get_settings

    get_settings.cache_clear()

    client, _ = _client_with_canned_answer(small_image)
    r = client.post(
        "/documents/upload",
        files={"file": ("test.png", small_image_bytes, "image/png")},
    )
    file_id = r.json()["file_id"]

    r2 = client.post(
        "/qa",
        json={
            "file_id": file_id,
            "question": "Extract fields",
            "output_mode": "json",
            "schema_name": "invoice_extraction",
            "backend": "mock",
            "route": True,
        },
    )
    assert r2.status_code == 200, r2.text
    body = r2.json()
    assert body["route_decision"] is not None
    assert body["route_decision"]["path"] in {"ocr", "small_vlm", "large_vlm"}

    log_path = tmp_path / "routing_log.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) >= 1


def test_qa_without_routing_does_not_create_log(
    small_image, small_image_bytes, tmp_path, monkeypatch
):
    monkeypatch.setenv("MMI_RESULTS_DIR", str(tmp_path))
    from app.config import get_settings

    get_settings.cache_clear()

    client, _ = _client_with_canned_answer(small_image)
    r = client.post(
        "/documents/upload",
        files={"file": ("test.png", small_image_bytes, "image/png")},
    )
    file_id = r.json()["file_id"]

    r2 = client.post(
        "/qa",
        json={
            "file_id": file_id,
            "question": "Extract fields",
            "output_mode": "json",
            "schema_name": "invoice_extraction",
            "backend": "mock",
            # route omitted → defaults to False
        },
    )
    assert r2.status_code == 200
    body = r2.json()
    assert body["route_decision"] is None
    assert not (tmp_path / "routing_log.jsonl").exists()
