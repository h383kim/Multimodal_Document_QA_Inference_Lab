"""POST /documents/upload — accepts image or PDF, returns file_id."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.api.deps import get_document_store
from app.ingestion.document_store import DocumentStore
from app.schemas.requests import UploadResponse

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    store: DocumentStore = Depends(get_document_store),
) -> UploadResponse:
    raw = await file.read()
    name = (file.filename or "").lower()
    content_type = (file.content_type or "").lower()
    is_pdf = name.endswith(".pdf") or content_type == "application/pdf"

    try:
        doc = store.add_pdf(raw) if is_pdf else store.add_image(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"could not parse upload: {exc}") from exc

    return UploadResponse(
        file_id=doc.file_id,
        num_pages=len(doc.pages),
        document_type=doc.document_type,
    )
