"""Adapters that map public document-QA datasets onto our JSONL row schema.

Every adapter is a pure transformer: it takes already-extracted fields from a
source dataset and returns the dict shape consumed by ``app.benchmarking.runner``::

    {"id": str, "file": str, "question": str, "expected": dict, "schema": str}

The image-saving and HuggingFace-loading concerns live in
``scripts/download_datasets.py`` so these functions stay testable without network
access or heavyweight dependencies.

Currently supported sources:
    - DocVQA (lmms-lab/DocVQA on HF) → schema "docvqa_answer"
    - CORD-v2 (naver-clova-ix/cord-v2) → schema "receipt_extraction"
    - SROIE (darentang/sroie or similar) → schema "receipt_extraction"
"""

from __future__ import annotations

from typing import Any, TypedDict


class JsonlRow(TypedDict):
    id: str
    file: str
    question: str
    expected: dict[str, Any]
    schema: str


_DEFAULT_DOCVQA_QUESTION_FALLBACK = "What is shown in the document?"
_RECEIPT_QUESTION = (
    "Extract the merchant name, purchase date, total amount, and payment method from this receipt."
)


def docvqa_row(
    item_id: str,
    file: str,
    question: str,
    answers: list[str] | None,
) -> JsonlRow:
    """Build a JSONL row from a DocVQA-style record.

    Picks the first non-empty answer as ground truth; DocVQA labels each question
    with a list of acceptable answers.
    """
    expected_answer = ""
    for ans in answers or []:
        if ans and ans.strip():
            expected_answer = ans.strip()
            break
    return {
        "id": item_id,
        "file": file,
        "question": question or _DEFAULT_DOCVQA_QUESTION_FALLBACK,
        "expected": {"answer": expected_answer},
        "schema": "docvqa_answer",
    }


def cord_row(item_id: str, file: str, ground_truth: dict[str, Any]) -> JsonlRow:
    """Build a JSONL row from a CORD-v2 ``ground_truth`` parse dict.

    CORD ground_truth is the parsed receipt JSON (the value under ``gt_parse``
    in the official release). Fields commonly present include ``menu``,
    ``sub_total``, ``total``, ``cashprice``, etc. We map onto our 4-field
    receipt schema and leave fields that CORD doesn't reliably carry as None.
    """
    parse = ground_truth.get("gt_parse", ground_truth) or {}
    total_block_raw = parse.get("total")
    total_block: dict[str, Any] = total_block_raw if isinstance(total_block_raw, dict) else {}
    total_amount = (
        total_block.get("total_price") or parse.get("total_price") or parse.get("cashprice")
    )
    return {
        "id": item_id,
        "file": file,
        "question": _RECEIPT_QUESTION,
        "expected": {
            "merchant_name": _first_str(parse, "store_name", "nm", "merchant"),
            "purchase_date": _first_str(parse, "date", "transaction_date"),
            "total_amount": _normalize_amount(total_amount),
            "payment_method": _first_str(parse, "payment_method", "tendertype"),
        },
        "schema": "receipt_extraction",
    }


def sroie_row(item_id: str, file: str, fields: dict[str, Any]) -> JsonlRow:
    """Build a JSONL row from SROIE's flat 4-key ground truth.

    SROIE labels are: ``company``, ``date``, ``address``, ``total``. Address
    isn't part of our receipt schema; payment_method isn't part of SROIE.
    """
    return {
        "id": item_id,
        "file": file,
        "question": _RECEIPT_QUESTION,
        "expected": {
            "merchant_name": _first_str(fields, "company", "merchant"),
            "purchase_date": _first_str(fields, "date"),
            "total_amount": _normalize_amount(fields.get("total")),
            "payment_method": None,
        },
        "schema": "receipt_extraction",
    }


def _first_str(source: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_amount(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    if isinstance(value, str):
        return value.strip() or None
    return None
