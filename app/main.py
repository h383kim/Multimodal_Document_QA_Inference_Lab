"""FastAPI app factory."""
from __future__ import annotations

from fastapi import FastAPI

from app import __version__
from app.api import routes_benchmark, routes_documents, routes_qa


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multimodal Document QA Inference Lab",
        version=__version__,
        description="Benchmark and optimization platform for multimodal document QA inference.",
    )
    app.include_router(routes_documents.router)
    app.include_router(routes_qa.router)
    app.include_router(routes_benchmark.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    return app


app = create_app()
