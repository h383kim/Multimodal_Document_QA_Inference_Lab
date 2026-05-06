"""Shared test fixtures."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.backends.mock_backend import MockBackend, _image_fingerprint


@pytest.fixture
def small_image() -> Image.Image:
    img = Image.new("RGB", (200, 120), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([(10, 10), (190, 110)], outline="black")
    draw.text((20, 40), "TEST", fill="black")
    return img


@pytest.fixture
def small_image_bytes(small_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    small_image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend(simulated_latency_ms=5.0)


@pytest.fixture
def fixture_dataset(tmp_path: Path) -> Path:
    """Build a tiny dataset on disk with two invoices and an answer book."""
    dataset = tmp_path / "fixture_invoices"
    images = dataset / "images"
    images.mkdir(parents=True)

    rows = []
    for i, expected in enumerate(
        [
            {
                "invoice_number": "INV-1",
                "vendor_name": "Vendor A",
                "due_date": "2026-01-01",
                "total_amount": "$10.00",
            },
            {
                "invoice_number": "INV-2",
                "vendor_name": "Vendor B",
                "due_date": "2026-02-02",
                "total_amount": "$20.00",
            },
        ],
        start=1,
    ):
        img = Image.new("RGB", (300, 200), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((20, 40), f"invoice {i}", fill="black")
        draw.text((20, 80), expected["invoice_number"], fill="black")
        rel = f"images/invoice_{i}.png"
        img.save(images / f"invoice_{i}.png")
        rows.append(
            {
                "id": f"invoice_{i}",
                "file": rel,
                "question": "Extract fields.",
                "expected": expected,
                "schema": "invoice_extraction",
            }
        )

    with (dataset / "labels.jsonl").open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return dataset


@pytest.fixture
def answer_book_for_fixture(fixture_dataset: Path) -> dict[str, str]:
    """Map every fixture image's fingerprint to a canned correct JSON answer."""
    book: dict[str, str] = {}
    for line in (fixture_dataset / "labels.jsonl").read_text().splitlines():
        row = json.loads(line)
        img = Image.open(fixture_dataset / row["file"]).convert("RGB")
        book[_image_fingerprint(img)] = json.dumps(row["expected"])
    return book
