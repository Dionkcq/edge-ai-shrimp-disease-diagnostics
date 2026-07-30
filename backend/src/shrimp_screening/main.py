"""The FastAPI application factory.

Everything that can fail is resolved here, before the first request: policy files
are parsed and hashed, the guidance corpus is validated against the lexicon, the
limitations document is parsed, and the detector is built. A misconfiguration is a
process that refuses to start, which is visible; a process that starts and then
returns 500s is not.

The one exception is the detector under the ``unavailable`` provider, which
"succeeds" by design and reports that no model is installed. That is the default
state of this repository and the correct answer for a clean checkout.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles

from shrimp_screening.api import routes_guidance, routes_health, routes_screening
from shrimp_screening.api.dependencies import RESOURCES_ATTRIBUTE, AppResources
from shrimp_screening.api.errors import (
    ApiProblemError,
    api_problem_handler,
    problem_response,
    unhandled_exception_handler,
)
from shrimp_screening.api.middleware import RequestContextMiddleware
from shrimp_screening.contracts.enums import ProblemCode
from shrimp_screening.detection.protocol import MarkerDetector
from shrimp_screening.detection.providers import build_detector
from shrimp_screening.guidance.store import load_guidance
from shrimp_screening.limitations import load_limitations
from shrimp_screening.paths import repository_root
from shrimp_screening.policy.loader import load_decision_policy, load_quality_policy
from shrimp_screening.settings import Settings, load_settings

API_TITLE = "Edge AI Shrimp Visible-Marker Screening"
API_VERSION = "0.1.0"

API_DESCRIPTION = (
    "Offline screening for two visible appearances in a shrimp photograph. This "
    "service reports appearances, never a diagnosis, and abstains rather than "
    "guessing. On a checkout with no trained weights the default provider is "
    "'unavailable': /readyz answers 503 and every screening answers "
    "UNABLE_TO_ASSESS / MODEL_UNAVAILABLE."
)

_logger = logging.getLogger("shrimp_screening")
_NON_SPA_PREFIXES = ("api", "assets", "livez", "readyz")


def build_resources(
    settings: Settings | None = None,
    *,
    detector: MarkerDetector | None = None,
) -> AppResources:
    """Load and validate every piece of process-wide state.

    ``detector`` is injectable so a test can exercise a provider without reaching
    through an environment variable. Nothing else about the wiring changes.
    """
    resolved = settings if settings is not None else load_settings()
    quality_policy = load_quality_policy()
    decision_policy = load_decision_policy()
    guidance = load_guidance()
    # Parsed at startup purely so a malformed document fails here rather than in
    # the middle of a response.
    load_limitations()
    return AppResources(
        settings=resolved,
        quality_policy=quality_policy,
        decision_policy=decision_policy,
        detector=detector if detector is not None else build_detector(resolved, decision_policy),
        guidance=guidance,
        inference_gate=asyncio.Semaphore(resolved.max_concurrent_inferences),
    )


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
    frontend_dir: Path | None = None,
) -> FastAPI:
    """Build the ASGI application."""
    resources = build_resources(settings, detector=detector)

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
    app.include_router(routes_screening.router)
    _mount_frontend(
        app,
        frontend_dir if frontend_dir is not None else repository_root() / "frontend" / "dist",
    )
    return app


def create_default_app() -> FastAPI:
    """Entry point for ``uvicorn shrimp_screening.main:create_default_app --factory``."""
    return create_app()
