"""OCR wrapper around pytesseract.

pytesseract is an optional dependency (install via the ``[ocr]`` extra) and
relies on a system-installed ``tesseract`` binary. ``run_ocr`` raises a clear
error when either is missing so callers can choose to fall back to a non-OCR
routing path.

The returned ``OCRResult`` exposes a normalized confidence in [0, 1] and the
extracted text. Callers (e.g. the router) can use this to decide whether a
pure-OCR path is viable.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from PIL.Image import Image


@dataclass
class OCRResult:
    text: str
    confidence: float  # 0..1, mean of per-word confidences
    word_count: int
    latency_ms: float
    backend: str = "pytesseract"


class OCRUnavailable(RuntimeError):
    """Raised when pytesseract or the tesseract binary isn't usable."""


def run_ocr(image: Image, lang: str = "eng") -> OCRResult:
    """Run OCR on a PIL image. Returns text + mean per-word confidence in [0, 1]."""
    try:
        import pytesseract
    except ImportError as exc:  # pragma: no cover — tested via stub elsewhere
        raise OCRUnavailable(
            "pytesseract is not installed. Install with: uv sync --extra ocr"
        ) from exc

    start = time.perf_counter()
    try:
        data: dict[str, Any] = pytesseract.image_to_data(
            image,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:  # pragma: no cover
        raise OCRUnavailable(
            "tesseract binary not found on PATH. Install via your package manager "
            "(e.g. `brew install tesseract` or `apt-get install tesseract-ocr`)."
        ) from exc

    confidences = [float(c) for c in data.get("conf", []) if c not in (None, -1, "-1")]
    words = [w for w in data.get("text", []) if w and w.strip()]
    text = " ".join(words)
    mean_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0
    return OCRResult(
        text=text,
        confidence=max(0.0, min(1.0, mean_conf)),
        word_count=len(words),
        latency_ms=(time.perf_counter() - start) * 1000.0,
    )
