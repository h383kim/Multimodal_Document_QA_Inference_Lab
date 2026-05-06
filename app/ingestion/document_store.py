"""In-memory document cache: file_id → pages.

MVP-only — replace with disk-backed storage when persistence is needed.
"""
from __future__ import annotations

import io
import uuid
from dataclasses import dataclass, field

from PIL import Image

from app.ingestion.image_preprocessor import resize_max_edge
from app.ingestion.pdf_loader import load_pdf_pages


@dataclass
class StoredDocument:
    file_id: str
    document_type: str  # "image" | "pdf"
    pages: list[Image.Image] = field(default_factory=list)


class DocumentStore:
    def __init__(self, max_image_edge: int = 1280) -> None:
        self._docs: dict[str, StoredDocument] = {}
        self.max_image_edge = max_image_edge

    def add_image(self, image_bytes: bytes) -> StoredDocument:
        img = Image.open(io.BytesIO(image_bytes))
        img = resize_max_edge(img, self.max_image_edge)
        return self._store("image", [img])

    def add_pdf(self, pdf_bytes: bytes, max_pages: int | None = None) -> StoredDocument:
        pages = load_pdf_pages(pdf_bytes, max_pages=max_pages)
        pages = [resize_max_edge(p, self.max_image_edge) for p in pages]
        return self._store("pdf", pages)

    def _store(self, document_type: str, pages: list[Image.Image]) -> StoredDocument:
        file_id = uuid.uuid4().hex[:12]
        doc = StoredDocument(file_id=file_id, document_type=document_type, pages=pages)
        self._docs[file_id] = doc
        return doc

    def get(self, file_id: str) -> StoredDocument | None:
        return self._docs.get(file_id)

    def add_image_object(self, image: Image.Image) -> StoredDocument:
        """Store a PIL.Image directly (used by tests and the benchmark runner)."""
        img = resize_max_edge(image, self.max_image_edge)
        return self._store("image", [img])
