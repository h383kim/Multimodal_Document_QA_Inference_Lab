"""Normalized field-level accuracy."""

from __future__ import annotations

import re
from typing import Any

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(value: Any) -> str:
    if value is None:
        return ""
    s = str(value).lower().strip()
    return _NON_ALNUM.sub("", s)


def field_match(predicted: Any, expected: Any) -> bool:
    """Exact-match after lowercase + non-alphanumeric removal.

    "$1,248.50" matches "1248.50"; "INV-001" matches "inv001"; whitespace insensitive.
    """
    return normalize(predicted) == normalize(expected)


def per_field_accuracy(predicted: dict[str, Any], expected: dict[str, Any]) -> dict[str, bool]:
    return {field: field_match(predicted.get(field), val) for field, val in expected.items()}


def record_accuracy(predicted: dict[str, Any] | None, expected: dict[str, Any]) -> float:
    """Fraction of fields that match. Returns 0.0 if predicted is None."""
    if predicted is None:
        return 0.0
    if not expected:
        return 1.0
    matches = sum(1 for ok in per_field_accuracy(predicted, expected).values() if ok)
    return matches / len(expected)
