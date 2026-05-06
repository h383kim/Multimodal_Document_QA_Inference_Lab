"""PDF → list[PIL.Image] via PyMuPDF."""
from __future__ import annotations

import io

import fitz
from PIL import Image


def load_pdf_pages(pdf_bytes: bytes, dpi: int = 150, max_pages: int | None = None) -> list[Image.Image]:
    """Render each page of a PDF to a PIL.Image at the given DPI."""
    images: list[Image.Image] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            if max_pages is not None and i >= max_pages:
                break
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
            images.append(img)
    return images
