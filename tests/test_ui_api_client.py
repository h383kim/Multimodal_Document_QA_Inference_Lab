"""Streamlit UI API client behavior."""

from __future__ import annotations

import json

import httpx
import pytest

from ui.api_client import ApiClientError, LabApiClient


def _client(handler):
    return LabApiClient(
        base_url="http://testserver",
        transport=httpx.MockTransport(handler),
    )


def test_health_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert request.url.path == "/health"
        return httpx.Response(200, json={"status": "ok", "version": "0.1.0"})

    assert _client(handler).health()["status"] == "ok"


def test_upload_document_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/documents/upload"
        body = request.read()
        assert b"invoice.png" in body
        return httpx.Response(
            200,
            json={"file_id": "abc123", "num_pages": 1, "document_type": "image"},
        )

    uploaded = _client(handler).upload_document("invoice.png", b"image-bytes", "image/png")
    assert uploaded.file_id == "abc123"
    assert uploaded.num_pages == 1
    assert uploaded.document_type == "image"


def test_ask_qa_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/qa"
        payload = json.loads(request.content)
        assert payload["file_id"] == "abc123"
        assert payload["question"] == "Extract fields"
        return httpx.Response(
            200,
            json={
                "answer": {"invoice_number": "INV-1"},
                "metrics": {
                    "ttft_ms": 1.0,
                    "total_latency_ms": 2.0,
                    "tokens_per_second": 3.0,
                    "peak_memory_mb": 0.0,
                    "retry_count": 0,
                    "schema_valid": True,
                },
                "backend": "mock",
                "model": "mock-vlm",
                "quantization": "none",
            },
        )

    response = _client(handler).ask_qa(file_id="abc123", question="Extract fields")
    assert response["answer"]["invoice_number"] == "INV-1"
    assert response["metrics"]["schema_valid"] is True


def test_error_detail_is_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "file_id not found"})

    with pytest.raises(ApiClientError) as exc_info:
        _client(handler).health()
    assert exc_info.value.status_code == 404
    assert "file_id not found" in str(exc_info.value)


def test_non_json_response_is_raised():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    with pytest.raises(ApiClientError, match="non-JSON"):
        _client(handler).health()
