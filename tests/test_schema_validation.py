"""JSON extraction + retry behaviour."""
from __future__ import annotations

from app.backends.mock_backend import MockBackend
from app.evals.schema_validation import extract_json, parse_json_with_retry
from app.schemas.outputs import InvoiceExtraction


def test_extract_bare_json():
    raw = '{"invoice_number": "INV-1"}'
    assert extract_json(raw) == raw


def test_extract_fenced_json():
    raw = "Here you go:\n```json\n{\"invoice_number\": \"INV-1\"}\n```\nThanks."
    extracted = extract_json(raw)
    assert extracted is not None
    assert "INV-1" in extracted


def test_extract_narrative_json():
    raw = "The answer is {\"invoice_number\": \"INV-1\", \"vendor_name\": \"Acme\"} hope this helps"
    extracted = extract_json(raw)
    assert extracted is not None
    assert "Acme" in extracted


def test_extract_returns_none_on_garbage():
    assert extract_json("there is no json here") is None
    assert extract_json("") is None


def test_parse_with_retry_happy_path(small_image):
    backend = MockBackend(simulated_latency_ms=1.0)
    backend.register_answer(small_image, '{"invoice_number": "INV-1"}')
    result = parse_json_with_retry(backend, small_image, "prompt", InvoiceExtraction, max_retries=2)
    assert result.schema_valid is True
    assert result.parsed.invoice_number == "INV-1"
    assert result.retry_count == 0


def test_parse_with_retry_normalizes_schema_keys(small_image):
    backend = MockBackend(simulated_latency_ms=1.0)
    backend.register_answer(
        small_image,
        '{"Invoice Number": "INV-1", "Vendor": "Acme", "Due Date": "2026-01-01", "Total Amount": "$5.00"}',
    )
    result = parse_json_with_retry(backend, small_image, "prompt", InvoiceExtraction, max_retries=2)
    assert result.schema_valid is True
    assert result.parsed.invoice_number == "INV-1"
    assert result.parsed.vendor_name == "Acme"
    assert result.parsed.due_date == "2026-01-01"
    assert result.parsed.total_amount == "$5.00"


def test_parse_with_retry_recovers_after_invalid_first_call(small_image):
    backend = MockBackend(simulated_latency_ms=1.0, force_invalid_first_call=True)
    backend.register_answer(small_image, '{"invoice_number": "INV-1"}')
    result = parse_json_with_retry(backend, small_image, "prompt", InvoiceExtraction, max_retries=2)
    assert result.schema_valid is True
    assert result.retry_count == 1


def test_parse_with_retry_gives_up(small_image):
    class AlwaysBad(MockBackend):
        def generate(self, image, prompt, generation_config=None):
            resp = super().generate(image, prompt, generation_config)
            resp["answer"] = "no json at all"
            return resp

    backend = AlwaysBad(simulated_latency_ms=1.0)
    result = parse_json_with_retry(backend, small_image, "prompt", InvoiceExtraction, max_retries=1)
    assert result.schema_valid is False
    assert result.parsed is None
    assert result.retry_count == 1
    assert result.parse_failure_reason is not None
