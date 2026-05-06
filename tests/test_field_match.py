"""Field normalization."""
from __future__ import annotations

from app.evals.field_match import field_match, per_field_accuracy, record_accuracy


def test_currency_normalized():
    assert field_match("$1,248.50", "1248.50")
    assert field_match("$1,248.50", "$1248.50")


def test_id_normalized():
    assert field_match("INV-001", "inv001")
    assert field_match("inv 001", "INV001")


def test_whitespace_insensitive():
    assert field_match("  Acme  Corp ", "acme corp")


def test_mismatch():
    assert not field_match("INV-001", "INV-002")
    assert not field_match(None, "x")


def test_per_field_accuracy():
    pred = {"a": "INV-1", "b": "$10"}
    exp = {"a": "inv1", "b": "$11"}
    out = per_field_accuracy(pred, exp)
    assert out == {"a": True, "b": False}


def test_record_accuracy():
    assert record_accuracy(None, {"a": "x"}) == 0.0
    assert record_accuracy({"a": "X"}, {"a": "x"}) == 1.0
    assert record_accuracy({"a": "X", "b": "Z"}, {"a": "x", "b": "Y"}) == 0.5
