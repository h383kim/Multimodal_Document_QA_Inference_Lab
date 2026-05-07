"""Routing policies. Start with a rule-based policy; the interface stays
narrow so a learned classifier can swap in later without changing callers."""

from __future__ import annotations

from typing import Protocol

from app.routing.types import RouteDecision, RouteFeatures

OCR_CONFIDENCE_THRESHOLD = 0.85
SIMPLE_IMAGE_PIXEL_LIMIT = 1500 * 2000  # ~3MP — single small page
SIMPLE_IMAGE_BYTE_LIMIT = 1_000_000  # 1MB

_OCR_FRIENDLY_SCHEMAS = {"invoice_extraction", "receipt_extraction"}


class RoutingPolicy(Protocol):
    def decide(self, features: RouteFeatures) -> RouteDecision: ...


class RuleBasedPolicy:
    """Three-tier escalation: ocr → small VLM → large VLM.

    Decision order:
        1. If the schema is OCR-friendly *and* OCR confidence clears the
           threshold and we have actual text, take the OCR path — fastest.
        2. Else if the image is "simple" (single page, modest pixel/bytes
           budget) and the question is not flagged complex, take small_vlm.
        3. Else fall through to large_vlm.
    """

    def __init__(
        self,
        ocr_confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
        simple_pixel_limit: int = SIMPLE_IMAGE_PIXEL_LIMIT,
        simple_byte_limit: int = SIMPLE_IMAGE_BYTE_LIMIT,
    ) -> None:
        self.ocr_confidence_threshold = ocr_confidence_threshold
        self.simple_pixel_limit = simple_pixel_limit
        self.simple_byte_limit = simple_byte_limit

    def decide(self, features: RouteFeatures) -> RouteDecision:
        feature_dict = {
            "image_pixels": features.image_pixels,
            "image_bytes": features.image_bytes,
            "is_multipage": features.is_multipage,
            "question_complexity": features.question_complexity,
            "schema_name": features.schema_name,
        }

        ocr_eligible = (
            features.schema_name in _OCR_FRIENDLY_SCHEMAS
            and features.has_ocr_text
            and features.ocr_confidence is not None
            and features.ocr_confidence >= self.ocr_confidence_threshold
        )
        if ocr_eligible:
            return RouteDecision(
                path="ocr",
                reason=(
                    f"ocr confidence {features.ocr_confidence:.2f} ≥ "
                    f"{self.ocr_confidence_threshold:.2f} on schema {features.schema_name!r}"
                ),
                ocr_confidence=features.ocr_confidence,
                features=feature_dict,
            )

        is_simple = (
            not features.is_multipage
            and features.image_pixels > 0
            and features.image_pixels <= self.simple_pixel_limit
            and features.image_bytes <= self.simple_byte_limit
            and features.question_complexity == "simple"
        )
        if is_simple:
            return RouteDecision(
                path="small_vlm",
                reason="single small page with simple question",
                ocr_confidence=features.ocr_confidence,
                features=feature_dict,
            )

        reason_parts: list[str] = []
        if features.is_multipage:
            reason_parts.append("multipage")
        if features.image_pixels > self.simple_pixel_limit:
            reason_parts.append(f"large image ({features.image_pixels} px)")
        if features.question_complexity == "complex":
            reason_parts.append("complex question")
        reason = "; ".join(reason_parts) or "fallback"
        return RouteDecision(
            path="large_vlm",
            reason=f"escalating to large VLM: {reason}",
            ocr_confidence=features.ocr_confidence,
            features=feature_dict,
        )
