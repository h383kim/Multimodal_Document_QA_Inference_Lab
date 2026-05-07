"""Document preview helper behavior."""
from __future__ import annotations

import io

import fitz
import pytest
from PIL import Image

from ui.document_preview import PreviewError, build_preview


def _png_bytes() -> bytes:
    image = Image.new("RGB", (40, 30), color="white")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _pdf_bytes() -> bytes:
    document = fitz.open()
    page = document.new_page(width=120, height=80)
    page.insert_text((20, 40), "invoice")
    return document.tobytes()


def test_build_image_preview():
    preview = build_preview(_png_bytes(), "invoice.png", "image/png")
    assert preview.kind == "image"
    assert preview.image.mode == "RGB"
    assert preview.image.size == (40, 30)


def test_build_pdf_preview_first_page():
    preview = build_preview(_pdf_bytes(), "invoice.pdf", "application/pdf")
    assert preview.kind == "pdf"
    assert preview.image.mode == "RGB"
    assert preview.image.width > 0
    assert preview.image.height > 0


def test_build_preview_rejects_invalid_image():
    with pytest.raises(PreviewError, match="could not preview image"):
        build_preview(b"not an image", "invoice.png", "image/png")


def test_build_preview_rejects_invalid_pdf():
    with pytest.raises(PreviewError, match="could not preview PDF"):
        build_preview(b"not a pdf", "invoice.pdf", "application/pdf")
