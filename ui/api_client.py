"""HTTP client for the FastAPI inference lab API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


class ApiClientError(RuntimeError):
    """Raised when the UI cannot complete an API request."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class UploadedDocument:
    file_id: str
    num_pages: int
    document_type: str


class LabApiClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 60.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def upload_document(
        self,
        filename: str,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> UploadedDocument:
        data = self._request(
            "POST",
            "/documents/upload",
            files={"file": (filename, content, content_type)},
        )
        return UploadedDocument(
            file_id=data["file_id"],
            num_pages=data["num_pages"],
            document_type=data["document_type"],
        )

    def ask_qa(
        self,
        *,
        file_id: str,
        question: str,
        output_mode: str = "json",
        schema_name: str | None = "invoice_extraction",
        backend: str | None = "mock",
        model: str | None = None,
        quantization: str | None = "fp16",
        max_retries: int = 2,
    ) -> dict[str, Any]:
        payload = {
            "file_id": file_id,
            "question": question,
            "output_mode": output_mode,
            "schema_name": schema_name,
            "backend": backend,
            "model": model or None,
            "quantization": quantization,
            "max_retries": max_retries,
        }
        return self._request("POST", "/qa", json=payload)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        try:
            with httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                transport=self.transport,
            ) as client:
                response = client.request(method, path, **kwargs)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise ApiClientError(
                _extract_error_detail(exc.response),
                status_code=exc.response.status_code,
            ) from exc
        except httpx.RequestError as exc:
            raise ApiClientError(f"could not connect to API at {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise ApiClientError("API returned a non-JSON response") from exc


def _extract_error_detail(response: httpx.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return response.text or f"HTTP {response.status_code}"
    detail = data.get("detail") if isinstance(data, dict) else None
    return str(detail or data or f"HTTP {response.status_code}")
