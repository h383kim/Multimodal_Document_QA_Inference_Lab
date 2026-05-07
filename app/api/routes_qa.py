"""POST /qa — answers a question about a previously-uploaded document."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_backend, get_document_store
from app.evals.schema_validation import parse_json_with_retry
from app.ingestion.document_store import DocumentStore
from app.schemas.outputs import get_schema
from app.schemas.requests import QAMetrics, QARequest, QAResponse


router = APIRouter(tags=["qa"])


def _build_prompt(question: str, schema_name: str | None, output_mode: str) -> str:
    base = f"You are a document understanding assistant. Question: {question}"
    if output_mode == "json" and schema_name:
        schema_cls = get_schema(schema_name)
        keys = list(schema_cls.model_fields.keys())
        return (
            f"{base}\n\nReply with a single JSON object containing exactly these keys: "
            f"{keys}. If a value is unknown, use null. Do not include prose."
        )
    return base


@router.post("/qa", response_model=QAResponse)
def qa(
    request: QARequest,
    store: DocumentStore = Depends(get_document_store),
) -> QAResponse:
    doc = store.get(request.file_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"file_id '{request.file_id}' not found")
    if not doc.pages:
        raise HTTPException(status_code=400, detail="document has no pages")

    try:
        backend = get_backend(request.backend, request.model, request.quantization)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"could not initialize backend '{request.backend or 'default'}': {exc}",
        ) from exc
    prompt = _build_prompt(request.question, request.schema_name, request.output_mode)

    if request.output_mode == "json":
        if not request.schema_name:
            raise HTTPException(status_code=400, detail="schema_name required for json mode")
        schema_cls = get_schema(request.schema_name)
        result = parse_json_with_retry(
            backend=backend,
            image=doc.pages[0],
            prompt=prompt,
            schema=schema_cls,
            max_retries=request.max_retries,
        )
        info = backend.get_model_info()
        answer = result.parsed.model_dump() if result.parsed else result.raw_answer
        backend_resp = result.last_backend_response
        return QAResponse(
            answer=answer,
            metrics=QAMetrics(
                ttft_ms=backend_resp["time_to_first_token_ms"],
                total_latency_ms=backend_resp["total_latency_ms"],
                tokens_per_second=backend_resp["tokens_per_second"],
                peak_memory_mb=backend_resp["peak_memory_mb"],
                retry_count=result.retry_count,
                schema_valid=result.schema_valid,
            ),
            backend=info["backend"],
            model=info["model"],
            quantization=info.get("quantization", "none"),
        )

    # natural_language / field_extraction → return raw text from a single call
    backend_resp = backend.generate(doc.pages[0], prompt)
    info = backend.get_model_info()
    return QAResponse(
        answer=backend_resp["answer"],
        metrics=QAMetrics(
            ttft_ms=backend_resp["time_to_first_token_ms"],
            total_latency_ms=backend_resp["total_latency_ms"],
            tokens_per_second=backend_resp["tokens_per_second"],
            peak_memory_mb=backend_resp["peak_memory_mb"],
            retry_count=0,
            schema_valid=True,
        ),
        backend=info["backend"],
        model=info["model"],
        quantization=info.get("quantization", "none"),
    )
