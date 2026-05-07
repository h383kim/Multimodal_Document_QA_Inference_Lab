"""POST /qa — answers a question about a previously-uploaded document."""

from __future__ import annotations

from functools import lru_cache

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.api.deps import get_backend, get_document_store
from app.config import get_settings
from app.evals.schema_validation import parse_json_with_retry
from app.ingestion.document_store import DocumentStore, StoredDocument
from app.routing.log import append_decision
from app.routing.router import Router
from app.routing.types import RouteDecision
from app.schemas.outputs import get_schema
from app.schemas.requests import QAMetrics, QARequest, QAResponse, RouteDecisionDTO

router = APIRouter(tags=["qa"])
log = structlog.get_logger("app.api.qa")


@lru_cache(maxsize=1)
def _get_router() -> Router:
    return Router()


def _route_request(doc: StoredDocument, request: QARequest) -> RouteDecision:
    decision = _get_router().route(
        image=doc.pages[0],
        question=request.question,
        schema_name=request.schema_name,
        is_multipage=len(doc.pages) > 1,
    )
    structlog_ctx = structlog.contextvars.get_contextvars()
    append_decision(
        get_settings().routing_log_path,
        decision,
        request_id=structlog_ctx.get("request_id"),
        schema_name=request.schema_name,
    )
    log.info(
        "qa.route",
        path=decision.path,
        reason=decision.reason,
        ocr_confidence=decision.ocr_confidence,
    )
    return decision


def _to_dto(decision: RouteDecision | None) -> RouteDecisionDTO | None:
    if decision is None:
        return None
    return RouteDecisionDTO(
        path=decision.path,
        reason=decision.reason,
        ocr_confidence=decision.ocr_confidence,
        features=decision.features,
    )


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

    route_decision = _route_request(doc, request) if request.route else None

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
        log.info(
            "qa.completed",
            mode="json",
            schema=request.schema_name,
            backend=info["backend"],
            model=info["model"],
            schema_valid=result.schema_valid,
            retry_count=result.retry_count,
            total_latency_ms=round(backend_resp["total_latency_ms"], 2),
        )
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
            route_decision=_to_dto(route_decision),
        )

    # natural_language / field_extraction → return raw text from a single call
    backend_resp = backend.generate(doc.pages[0], prompt)
    info = backend.get_model_info()
    log.info(
        "qa.completed",
        mode=request.output_mode,
        backend=info["backend"],
        model=info["model"],
        total_latency_ms=round(backend_resp["total_latency_ms"], 2),
    )
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
        route_decision=_to_dto(route_decision),
    )
