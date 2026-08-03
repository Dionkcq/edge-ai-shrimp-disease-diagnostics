"""``GET /api/v1/advice/{decision}`` -- optional, local, generated advice.

This endpoint exists behind ``SHRIMP_LLM_ENABLED`` (default off) because it adds a
dependency this repository does not otherwise have: a running local Ollama
server. When the feature is off, or Ollama cannot be reached, or its answer
fails the safety scan, this endpoint fails the same way the rest of this API
fails elsewhere -- explicitly, with a stable problem code -- rather than serving
a partial or silently degraded answer.

The cited, non-generative guidance at ``GET /api/v1/guidance/{decision}`` always
works regardless of this endpoint, and is what this endpoint expands on, never
replaces.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from shrimp_screening.api.dependencies import AppResources, get_resources
from shrimp_screening.api.errors import ApiProblemError
from shrimp_screening.contracts.advice import AdviceDocument
from shrimp_screening.contracts.enums import Decision, ProblemCode
from shrimp_screening.contracts.guidance import CitedSource
from shrimp_screening.limitations import limitation_ids_for
from shrimp_screening.llm.advisor import REVIEW_NOTE, AdviceGenerationError, generate_advice

router = APIRouter(prefix="/api/v1", tags=["advice"])

Resources = Annotated[AppResources, Depends(get_resources)]


@router.get("/advice/{decision}", response_model=AdviceDocument)
async def advice_for_decision(decision: str, resources: Resources) -> AdviceDocument:
    """Return locally generated advice for one decision, or fail explicitly.

    The path parameter is typed as ``str``, matching ``routes_guidance``, so an
    unknown decision produces this module's stable ``NOT_FOUND`` body rather than
    FastAPI's generic 422 validation envelope.
    """
    try:
        member = Decision(decision)
    except ValueError as exc:
        raise ApiProblemError(
            ProblemCode.NOT_FOUND,
            404,
            "No guidance exists for that decision.",
        ) from exc

    if resources.llm_client is None:
        raise ApiProblemError(
            ProblemCode.NOT_FOUND,
            404,
            "Local advice generation is not enabled on this build.",
        )

    guidance_item = resources.guidance.for_decision(member)
    try:
        content = await generate_advice(
            resources.llm_client, decision=member, guidance_item=guidance_item
        )
    except AdviceGenerationError as exc:
        raise ApiProblemError(
            ProblemCode.ADVICE_UNAVAILABLE,
            503,
            "Local advice generation is temporarily unavailable.",
            retry_after_seconds=resources.settings.retry_after_seconds,
        ) from exc

    return AdviceDocument(
        decision=member,
        summary=content.summary,
        immediate_actions=list(content.immediate_actions),
        prevention_actions=list(content.prevention_actions),
        additional_considerations=list(content.additional_considerations),
        based_on_guidance_id=guidance_item.guidance_id,
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
        model_id=resources.settings.llm_model,
        review_note=REVIEW_NOTE,
        limitations=limitation_ids_for(member),
    )
