"""FastAPI app factory."""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import Response

from app import __version__
from app.api import routes_benchmark, routes_documents, routes_qa
from app.config import get_settings
from app.logging import configure_logging

REQUEST_ID_HEADER = "X-Request-ID"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    log = structlog.get_logger("app.main")
    log.info(
        "app.startup",
        version=__version__,
        default_backend=settings.default_backend,
        log_level=settings.log_level,
    )
    yield
    log.info("app.shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Multimodal Document QA Inference Lab",
        version=__version__,
        description="Benchmark and optimization platform for multimodal document QA inference.",
        lifespan=lifespan,
    )

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
        )
        log = structlog.get_logger("app.http")
        start = time.perf_counter()
        try:
            response: Response = await call_next(request)
        except Exception:
            log.exception("http.request_failed")
            raise
        duration_ms = (time.perf_counter() - start) * 1000.0
        log.info(
            "http.request",
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    app.include_router(routes_documents.router)
    app.include_router(routes_qa.router)
    app.include_router(routes_benchmark.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
