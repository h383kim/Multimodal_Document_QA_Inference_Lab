"""Router: turns a (image, question, schema) tuple into a RouteDecision.

The router gathers cheap features (image dimensions, OCR confidence, question
keywords) and hands them to a policy. The policy returns a RouteDecision —
the dispatch (which backend actually handles the request) is the caller's
responsibility, so policies can be replaced (rule-based → classifier) without
touching the QA route.
"""

from __future__ import annotations

import io
from collections.abc import Callable

from PIL.Image import Image

from app.ingestion.ocr import OCRResult, OCRUnavailable, run_ocr
from app.routing.policies import RoutingPolicy, RuleBasedPolicy
from app.routing.types import RouteDecision, RouteFeatures

_COMPLEX_KEYWORDS = (
    "summarize",
    "summary",
    "explain",
    "describe",
    "compare",
    "analyze",
    "interpret",
    "why",
)


def _classify_question(question: str) -> str:
    lowered = question.lower()
    for keyword in _COMPLEX_KEYWORDS:
        if keyword in lowered:
            return "complex"
    return "simple"


def _image_byte_estimate(image: Image) -> int:
    """Rough byte budget: re-encode as PNG in memory. Cheap for small images."""
    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return buf.tell()


class Router:
    def __init__(
        self,
        policy: RoutingPolicy | None = None,
        ocr_runner: Callable[[Image], OCRResult] | None = None,
        run_ocr_on_route: bool = True,
    ) -> None:
        self.policy = policy or RuleBasedPolicy()
        self._ocr_runner = ocr_runner or run_ocr
        self.run_ocr_on_route = run_ocr_on_route

    def route(
        self,
        image: Image,
        question: str,
        schema_name: str | None,
        is_multipage: bool = False,
    ) -> RouteDecision:
        ocr_confidence: float | None = None
        ocr_text = ""
        if self.run_ocr_on_route:
            try:
                result = self._ocr_runner(image)
                ocr_confidence = result.confidence
                ocr_text = result.text
            except OCRUnavailable:
                ocr_confidence = None
                ocr_text = ""

        width, height = image.size
        features = RouteFeatures(
            ocr_confidence=ocr_confidence,
            has_ocr_text=bool(ocr_text.strip()),
            image_pixels=width * height,
            image_bytes=_image_byte_estimate(image),
            is_multipage=is_multipage,
            question_complexity=_classify_question(question),  # type: ignore[arg-type]
            schema_name=schema_name,
        )
        return self.policy.decide(features)
