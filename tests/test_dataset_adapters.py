"""Unit tests for the public-dataset adapters.

These tests pass synthetic in-memory records (the kind a HuggingFace
``datasets.load_dataset`` row would produce, after raw-JSON parsing) and assert
the JSONL row schema produced for each source.
"""

from __future__ import annotations

from app.ingestion.dataset_adapters import cord_row, docvqa_row, sroie_row


def test_docvqa_row_picks_first_non_empty_answer():
    row = docvqa_row(
        item_id="docvqa_42",
        file="images/docvqa_42.png",
        question="What is the invoice number?",
        answers=["", "  ", "INV-9000", "INV-9000"],
    )
    assert row == {
        "id": "docvqa_42",
        "file": "images/docvqa_42.png",
        "question": "What is the invoice number?",
        "expected": {"answer": "INV-9000"},
        "schema": "docvqa_answer",
    }


def test_docvqa_row_empty_answers_falls_back_to_empty_string():
    row = docvqa_row(item_id="x", file="f.png", question="Q?", answers=None)
    assert row["expected"] == {"answer": ""}


def test_docvqa_row_blank_question_uses_fallback():
    row = docvqa_row(item_id="x", file="f.png", question="", answers=["y"])
    assert row["question"]  # non-empty fallback


def test_cord_row_extracts_known_fields():
    ground_truth = {
        "gt_parse": {
            "store_name": "Cafe Lumiere",
            "date": "2026-04-15",
            "total": {"total_price": "$23.40"},
            "payment_method": "card",
        }
    }
    row = cord_row(item_id="cord_1", file="images/cord_1.png", ground_truth=ground_truth)
    assert row["schema"] == "receipt_extraction"
    assert row["expected"] == {
        "merchant_name": "Cafe Lumiere",
        "purchase_date": "2026-04-15",
        "total_amount": "$23.40",
        "payment_method": "card",
    }


def test_cord_row_handles_flat_dict_without_gt_parse_wrapper():
    row = cord_row(
        item_id="cord_2",
        file="images/cord_2.png",
        ground_truth={"nm": "QuickMart", "date": "2026-01-01", "total_price": 12.5},
    )
    assert row["expected"]["merchant_name"] == "QuickMart"
    assert row["expected"]["total_amount"] == "12.50"


def test_cord_row_missing_fields_are_none():
    row = cord_row(item_id="cord_3", file="images/cord_3.png", ground_truth={})
    assert row["expected"] == {
        "merchant_name": None,
        "purchase_date": None,
        "total_amount": None,
        "payment_method": None,
    }


def test_sroie_row_maps_canonical_four_fields():
    row = sroie_row(
        item_id="sroie_1",
        file="images/sroie_1.png",
        fields={
            "company": "Soylent Foods",
            "date": "2026-05-22",
            "address": "1 Market St",
            "total": "67.89",
        },
    )
    assert row["expected"] == {
        "merchant_name": "Soylent Foods",
        "purchase_date": "2026-05-22",
        "total_amount": "67.89",
        "payment_method": None,
    }


def test_sroie_row_uses_receipt_extraction_schema():
    row = sroie_row(item_id="x", file="f.png", fields={})
    assert row["schema"] == "receipt_extraction"
