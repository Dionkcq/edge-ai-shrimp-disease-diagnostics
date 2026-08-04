"""Liveness, readiness and service metadata.

The separation between the two health endpoints is the fix for a specific failure
mode: if a missing model reads as a normal product state, then a broken deployment
looks healthy and the whole test suite stays vacuously green.

* ``/livez``  -- always 200 while the process is running.
* ``/readyz`` -- **503 when no detector is loaded.** On a clean checkout, which has
  no weights, this is the correct answer and there is a test asserting it.
"""

from __future__ import annotations

from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from shrimp_screening.contracts.enums import Decision, ProviderKind
from shrimp_screening.contracts.screening import SCHEMA_VERSION
from shrimp_screening.resources import AppResources
from shrimp_server.dependencies import get_resources

router = APIRouter(tags=["health"])

Resources = Annotated[AppResources, Depends(get_resources)]

_STRICT = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())


class LivenessResponse(BaseModel):
    model_config = _STRICT

    status: Literal["alive"] = "alive"


class ReadinessResponse(BaseModel):
    model_config = _STRICT

    status: Literal["ready", "not_ready"]
    provider: ProviderKind
    model_available: bool
    reason: str | None = Field(
        default=None,
        description="Why the service is not ready. Null when it is.",
    )


class MetaResponse(BaseModel):
    """Static facts a client needs before it can render anything."""

    model_config = _STRICT

    service: Literal["shrimp-marker-screening"] = "shrimp-marker-screening"
    schema_version: str
    environment: str
    provider: ProviderKind
    model_available: bool
    is_demonstration_data: bool
    decisions: list[Decision]
    quality_policy_id: str
    quality_policy_hash: str
    decision_policy_id: str
    decision_policy_hash: str
    guidance_review_status: str
    max_upload_bytes: int
    #: Whether ``GET /api/v1/advice/{decision}`` exists on this build. False means
    #: that endpoint answers 404, and a client must not offer the feature at all
    #: rather than discovering it by asking and failing.
    advice_available: bool
    chat_available: bool
    offline: Literal[True] = True


@router.get("/livez", response_model=LivenessResponse)
def livez() -> LivenessResponse:
    return LivenessResponse()


@router.get("/readyz", response_model=ReadinessResponse)
def readyz(resources: Resources) -> JSONResponse:
    metadata = resources.detector.metadata
    ready = metadata.available
    body = ReadinessResponse(
        status="ready" if ready else "not_ready",
        provider=metadata.provider,
        model_available=metadata.available,
        reason=None if ready else "MODEL_UNAVAILABLE",
    )
    return JSONResponse(body.model_dump(mode="json"), status_code=200 if ready else 503)


@router.get("/api/v1/meta", response_model=MetaResponse)
def meta(resources: Resources) -> MetaResponse:
    metadata = resources.detector.metadata
    return MetaResponse(
        schema_version=SCHEMA_VERSION,
        environment=resources.settings.env,
        provider=metadata.provider,
        model_available=metadata.available,
        is_demonstration_data=metadata.demonstration,
        decisions=list(Decision),
        quality_policy_id=resources.quality_policy.policy_id,
        quality_policy_hash=resources.quality_policy.policy_hash,
        decision_policy_id=resources.decision_policy.policy_id,
        decision_policy_hash=resources.decision_policy.policy_hash,
        guidance_review_status=resources.guidance.review_status,
        max_upload_bytes=resources.settings.max_upload_bytes,
        advice_available=resources.llm_client is not None,
        chat_available=resources.llm_client is not None,
    )
