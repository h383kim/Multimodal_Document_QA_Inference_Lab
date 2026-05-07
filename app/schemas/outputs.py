"""Pydantic models for structured QA outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class InvoiceExtraction(BaseModel):
    invoice_number: str | None = Field(default=None)
    vendor_name: str | None = Field(default=None)
    due_date: str | None = Field(default=None)
    total_amount: str | None = Field(default=None)


class ReceiptExtraction(BaseModel):
    merchant_name: str | None = Field(default=None)
    purchase_date: str | None = Field(default=None)
    total_amount: str | None = Field(default=None)
    payment_method: str | None = Field(default=None)


class DocVQAAnswer(BaseModel):
    answer: str | None = Field(default=None)


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {
    "invoice_extraction": InvoiceExtraction,
    "receipt_extraction": ReceiptExtraction,
    "docvqa_answer": DocVQAAnswer,
}


def get_schema(name: str) -> type[BaseModel]:
    if name not in SCHEMA_REGISTRY:
        raise KeyError(f"unknown schema '{name}'. Known: {sorted(SCHEMA_REGISTRY)}")
    return SCHEMA_REGISTRY[name]
