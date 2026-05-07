"""Routing policy + Router orchestration tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from app.ingestion.ocr import OCRResult, OCRUnavailable
from app.routing.log import append_decision
from app.routing.policies import RuleBasedPolicy
from app.routing.router import Router
from app.routing.types import RouteDecision, RouteFeatures

# ---- RuleBasedPolicy ----


def _features(**overrides):
    base = RouteFeatures(
        ocr_confidence=None,
        has_ocr_text=False,
        image_pixels=400 * 400,
        image_bytes=50_000,
        is_multipage=False,
        question_complexity="simple",
        schema_name="invoice_extraction",
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_policy_picks_ocr_when_confidence_high_on_ocr_friendly_schema():
    policy = RuleBasedPolicy()
    decision = policy.decide(_features(ocr_confidence=0.92, has_ocr_text=True))
    assert decision.path == "ocr"
    assert "ocr confidence" in decision.reason
    assert decision.ocr_confidence == 0.92


def test_policy_skips_ocr_for_unfriendly_schema_even_with_high_confidence():
    policy = RuleBasedPolicy()
    decision = policy.decide(
        _features(
            ocr_confidence=0.95,
            has_ocr_text=True,
            schema_name="docvqa_answer",
        )
    )
    assert decision.path == "small_vlm"


def test_policy_skips_ocr_when_confidence_below_threshold():
    policy = RuleBasedPolicy()
    decision = policy.decide(_features(ocr_confidence=0.6, has_ocr_text=True))
    assert decision.path == "small_vlm"


def test_policy_picks_small_vlm_for_simple_image_and_question():
    policy = RuleBasedPolicy()
    decision = policy.decide(_features())
    assert decision.path == "small_vlm"


def test_policy_escalates_to_large_vlm_for_complex_question():
    policy = RuleBasedPolicy()
    decision = policy.decide(_features(question_complexity="complex"))
    assert decision.path == "large_vlm"
    assert "complex question" in decision.reason


def test_policy_escalates_for_multipage():
    policy = RuleBasedPolicy()
    decision = policy.decide(_features(is_multipage=True))
    assert decision.path == "large_vlm"
    assert "multipage" in decision.reason


def test_policy_escalates_for_oversized_image():
    policy = RuleBasedPolicy()
    decision = policy.decide(_features(image_pixels=4000 * 4000))
    assert decision.path == "large_vlm"


# ---- Router orchestration ----


def _ocr_stub(confidence: float, text: str = "Invoice INV-100"):
    def runner(_image):
        return OCRResult(text=text, confidence=confidence, word_count=2, latency_ms=1.0)

    return runner


def test_router_uses_ocr_runner_to_extract_confidence():
    image = Image.new("RGB", (300, 300), color="white")
    router = Router(ocr_runner=_ocr_stub(0.9))
    decision = router.route(image, "Extract invoice number", "invoice_extraction")
    assert decision.path == "ocr"
    assert decision.ocr_confidence == 0.9


def test_router_falls_through_when_ocr_unavailable():
    image = Image.new("RGB", (300, 300), color="white")

    def unavailable(_image):
        raise OCRUnavailable("test")

    router = Router(ocr_runner=unavailable)
    decision = router.route(image, "Extract invoice number", "invoice_extraction")
    assert decision.path == "small_vlm"
    assert decision.ocr_confidence is None


def test_router_classifies_summary_question_as_complex():
    image = Image.new("RGB", (300, 300), color="white")
    router = Router(ocr_runner=_ocr_stub(0.9), run_ocr_on_route=False)
    decision = router.route(image, "Summarize this document", "invoice_extraction")
    assert decision.path == "large_vlm"


def test_router_skips_ocr_when_disabled():
    image = Image.new("RGB", (300, 300), color="white")
    router = Router(run_ocr_on_route=False)
    decision = router.route(image, "What is this?", "invoice_extraction")
    # OCR not called → no confidence — falls through to small_vlm
    assert decision.path == "small_vlm"
    assert decision.ocr_confidence is None


# ---- Routing log ----


def test_append_decision_writes_jsonl(tmp_path: Path):
    log_path = tmp_path / "routing_log.jsonl"
    decision = RouteDecision(
        path="ocr",
        reason="test",
        ocr_confidence=0.9,
        features={"image_pixels": 90000},
    )

    append_decision(log_path, decision, request_id="req-1", schema_name="invoice_extraction")
    append_decision(log_path, decision, request_id="req-2", schema_name="invoice_extraction")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    import json

    record = json.loads(lines[0])
    assert record["path"] == "ocr"
    assert record["request_id"] == "req-1"
    assert record["schema_name"] == "invoice_extraction"
    assert "timestamp" in record


def test_append_decision_creates_parent_dir(tmp_path: Path):
    log_path = tmp_path / "nested" / "deeper" / "routing.jsonl"
    decision = RouteDecision(path="small_vlm", reason="x")
    append_decision(log_path, decision)
    assert log_path.exists()


def test_ocr_unavailable_when_pytesseract_missing(monkeypatch):
    """If pytesseract isn't importable, run_ocr should raise OCRUnavailable."""
    import sys

    from app.ingestion import ocr

    monkeypatch.setitem(sys.modules, "pytesseract", None)
    image = Image.new("RGB", (10, 10))
    with pytest.raises(OCRUnavailable):
        ocr.run_ocr(image)
