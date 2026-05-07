"""API behavior when backend initialization fails."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.deps import _build_backend
from app.main import create_app


def test_qa_backend_initialization_error_is_reported(monkeypatch, small_image):
    app = create_app()
    client = TestClient(app)
    _build_backend.cache_clear()

    store = app.dependency_overrides
    assert store == {}

    upload = client.post(
        "/documents/upload",
        files={"file": ("test.png", _png_bytes(small_image), "image/png")},
    )
    assert upload.status_code == 200

    def fail_backend(*_args, **_kwargs):
        raise ImportError("missing dependency")

    import app.api.routes_qa as routes_qa

    monkeypatch.setattr(routes_qa, "get_backend", fail_backend)

    response = client.post(
        "/qa",
        json={
            "file_id": upload.json()["file_id"],
            "question": "Extract fields",
            "output_mode": "json",
            "schema_name": "invoice_extraction",
            "backend": "transformers",
        },
    )

    assert response.status_code == 500
    assert "could not initialize backend 'transformers'" in response.json()["detail"]
    assert "missing dependency" in response.json()["detail"]


def _png_bytes(image):
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
