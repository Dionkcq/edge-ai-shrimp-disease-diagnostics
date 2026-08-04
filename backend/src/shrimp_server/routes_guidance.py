"""Guidance retrieval: one deterministic lookup per decision."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from shrimp_screening.contracts.enums import Decision, ProblemCode
from shrimp_screening.contracts.guidance import CitedSource, GuidanceDocument
from shrimp_screening.limitations import limitation_ids_for
from shrimp_screening.problems import ApiProblemError
from shrimp_screening.resources import AppResources
from shrimp_server.dependencies import get_resources

router = APIRouter(prefix="/api/v1", tags=["guidance"])

Resources = Annotated[AppResources, Depends(get_resources)]


@router.get("/guidance/{decision}", response_model=GuidanceDocument)
def guidance_for_decision(decision: str, resources: Resources) -> GuidanceDocument:
    """Return the cited guidance for one decision.

    The path parameter is typed as ``str`` rather than ``Decision`` on purpose: an
    unknown decision must produce this module's problem+json body with a stable
    ``NOT_FOUND`` code, not FastAPI's generic 422 validation envelope.
    """
    try:
        member = Decision(decision)
    except ValueError as exc:
        raise ApiProblemError(
            ProblemCode.NOT_FOUND,
            404,
            "No guidance exists for that decision.",
        ) from exc

    item = resources.guidance.for_decision(member)
    return GuidanceDocument(
        decision=member,
        id=item.guidance_id,
        headline=item.headline,
        body=item.body,
        sources=[
            CitedSource(
                id=source.source_id,
                title=source.title,
                publisher=source.publisher,
                url=source.url,
                accessed_on=source.accessed_on,
            )
            for source in resources.guidance.citations_for(member)
        ],
        review_status=resources.guidance.review_status,
        review_note=resources.guidance.review_note,
        limitations=limitation_ids_for(member),
    )
