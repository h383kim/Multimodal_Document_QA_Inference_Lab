"""FastAPI dependency helper behavior."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.api.deps import _register_sample_invoice_answers
from app.backends.mock_backend import MockBackend


def test_register_sample_invoice_answers(tmp_path: Path):
    dataset = tmp_path / "sample_invoices"
    images = dataset / "images"
    images.mkdir(parents=True)

    image = Image.new("RGB", (120, 80), color="white")
    image_path = images / "invoice.png"
    image.save(image_path)

    expected = {
        "invoice_number": "INV-TEST",
        "vendor_name": "Test Vendor",
        "due_date": "2026-01-01",
        "total_amount": "$10.00",
    }
    labels = dataset / "labels.jsonl"
    labels.write_text(
        json.dumps(
            {
                "id": "invoice",
                "file": "images/invoice.png",
                "question": "Extract fields.",
                "expected": expected,
                "schema": "invoice_extraction",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    backend = MockBackend(simulated_latency_ms=0)
    registered = _register_sample_invoice_answers(backend, dataset)

    response = backend.generate(image, "extract")
    assert registered == 1
    assert json.loads(response["answer"]) == expected


def test_register_sample_invoice_answers_missing_dataset(tmp_path: Path):
    backend = MockBackend(simulated_latency_ms=0)
    assert _register_sample_invoice_answers(backend, tmp_path / "missing") == 0
