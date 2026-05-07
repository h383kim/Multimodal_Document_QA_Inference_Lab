"""Routing types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RoutePath = Literal["ocr", "small_vlm", "large_vlm"]


@dataclass
class RouteFeatures:
    """Inputs the policy uses to decide a path."""

    ocr_confidence: float | None = None
    has_ocr_text: bool = False
    image_pixels: int = 0
    image_bytes: int = 0
    is_multipage: bool = False
    question_complexity: Literal["simple", "complex"] = "simple"
    schema_name: str | None = None


@dataclass
class RouteDecision:
    path: RoutePath
    reason: str
    ocr_confidence: float | None = None
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "reason": self.reason,
            "ocr_confidence": self.ocr_confidence,
            "features": self.features,
        }
