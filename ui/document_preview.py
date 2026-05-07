"""Local document preview helpers for the Streamlit UI."""
from __future__ import annotations

import io
from dataclasses import dataclass

import fitz
from PIL import Image, UnidentifiedImageError


class PreviewError(RuntimeError):
    """Raised when a local preview image cannot be created."""


@dataclass(frozen=True)
class PreviewDocument:
    image: Image.Image
    kind: str
    label: str


def build_preview(
    file_bytes: bytes,
    filename: str,
    content_type: str = "",
) -> PreviewDocument:
    name = filename.lower()
    content_type = content_type.lower()
    is_pdf = name.endswith(".pdf") or content_type == "application/pdf"
    if is_pdf:
        return _build_pdf_preview(file_bytes)
    return _build_image_preview(file_bytes)


def _build_image_preview(file_bytes: bytes) -> PreviewDocument:
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise PreviewError(f"could not preview image: {exc}") from exc
    return PreviewDocument(image=image, kind="image", label="Image preview")


def _build_pdf_preview(file_bytes: bytes) -> PreviewDocument:
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            if document.page_count == 0:
                raise PreviewError("PDF has no pages")
            page = document[0]
            matrix = fitz.Matrix(150 / 72.0, 150 / 72.0)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    except PreviewError:
        raise
    except Exception as exc:
        raise PreviewError(f"could not preview PDF: {exc}") from exc
    return PreviewDocument(image=image, kind="pdf", label="PDF page 1 preview")
