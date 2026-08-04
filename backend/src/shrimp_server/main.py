"""The FastAPI application factory.

This module is the HTTP surface and nothing else: it wires routers, middleware and
exception handlers onto an app, and serves the built frontend. The state those
routes read is loaded and validated by
:func:`shrimp_screening.resources.build_resources`, which runs before the first
request, so a misconfiguration is a process that refuses to start -- which is
visible -- rather than a process that starts and then returns 500s.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from shrimp_screening.contracts.enums import ProblemCode
from shrimp_screening.detection.protocol import MarkerDetector
from shrimp_screening.llm.client import OllamaClient
from shrimp_screening.paths import repository_root
from shrimp_screening.problems import ApiProblemError
from shrimp_screening.resources import build_resources
from shrimp_screening.settings import API_DESCRIPTION, API_TITLE, API_VERSION, Settings
from shrimp_server import (
    routes_advice,
    routes_chat,
    routes_guidance,
    routes_health,
    routes_screening,
)
from shrimp_server.dependencies import RESOURCES_ATTRIBUTE
from shrimp_server.errors import (
    api_problem_handler,
    problem_response,
    unhandled_exception_handler,
)
from shrimp_server.middleware import RequestContextMiddleware

_logger = logging.getLogger("shrimp_screening")
_NON_SPA_PREFIXES = ("api", "assets", "livez", "readyz")


async def _not_found_handler(request: Request, exc: Exception) -> JSONResponse:
    del exc
    return problem_response(
        ApiProblemError(ProblemCode.NOT_FOUND, 404, "No such resource."),
        getattr(request.state, "request_id", None),
    )


def _mount_frontend(app: FastAPI, frontend_dir: Path) -> None:
    """Serve a built Vite client without allowing SPA fallback to hide API errors."""
    index = frontend_dir / "index.html"
    if not index.is_file():
        _logger.info("frontend build not mounted: %s is absent", index)
        return

    assets = frontend_dir / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="frontend-assets")

    async def serve_index(frontend_path: str = "") -> FileResponse:
        first_segment = frontend_path.partition("/")[0]
        if first_segment in _NON_SPA_PREFIXES:
            raise ApiProblemError(ProblemCode.NOT_FOUND, 404, "No such resource.")
        return FileResponse(index, media_type="text/html", headers={"Cache-Control": "no-cache"})

    app.add_api_route("/", serve_index, methods=["GET", "HEAD"], include_in_schema=False)
    app.add_api_route(
        "/{frontend_path:path}",
        serve_index,
        methods=["GET", "HEAD"],
        include_in_schema=False,
    )


def create_app(
    settings: Settings | None = None,
    *,
    detector: MarkerDetector | None = None,
    llm_client: OllamaClient | None = None,
    frontend_dir: Path | None = None,
) -> FastAPI:
    """Build the ASGI application."""
    resources = build_resources(settings, detector=detector, llm_client=llm_client)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.resources = resources
        metadata = resources.detector.metadata
        _logger.info(
            "starting env=%s provider=%s model_available=%s",
            resources.settings.env,
            metadata.provider.value,
            metadata.available,
        )
        yield

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=lifespan,
        docs_url="/api/docs",
        redoc_url=None,
        openapi_url="/api/openapi.json",
    )
    # Set eagerly as well as in the lifespan, so a caller that mounts this app
    # without running startup still resolves its dependencies.
    setattr(app.state, RESOURCES_ATTRIBUTE, resources)

    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(ApiProblemError, api_problem_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
    app.add_exception_handler(404, _not_found_handler)

    app.include_router(routes_health.router)
    app.include_router(routes_guidance.router)
    app.include_router(routes_advice.router)
    app.include_router(routes_chat.router)
    app.include_router(routes_screening.router)
    _mount_frontend(
        app,
        frontend_dir if frontend_dir is not None else repository_root() / "frontend" / "dist",
    )
    return app


def create_default_app() -> FastAPI:
    """Entry point for ``uvicorn shrimp_server.main:create_default_app --factory``."""
    return create_app()
