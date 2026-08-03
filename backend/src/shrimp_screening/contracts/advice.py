"""The response contract for locally generated, unreviewed advice.

``review_status`` is a fixed literal, not a free string, for the same reason
:class:`~shrimp_screening.contracts.guidance.GuidanceDocument` fixes its own
review status: there is no field arrangement in which a client can render this
advice while omitting the fact that a language model wrote it and nobody
reviewed it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from shrimp_screening.contracts.enums import Decision
from shrimp_screening.contracts.guidance import CitedSource

_STRICT = ConfigDict(extra="forbid", frozen=True)


class AdviceDocument(BaseModel):
    """Response body of ``GET /api/v1/advice/{decision}``."""

    model_config = _STRICT

    decision: Decision
    summary: str = Field(min_length=1)
    immediate_actions: list[str] = Field(min_length=1)
    prevention_actions: list[str] = Field(min_length=1)
    additional_considerations: list[str] = Field(default_factory=list)
    #: The cited guidance item this advice was expanded from. Traceable back to
    #: `GET /api/v1/guidance/{decision}`, which returns the same id.
    based_on_guidance_id: str = Field(min_length=1)
    sources: list[CitedSource] = Field(min_length=1)
    provider: Literal["ollama"] = "ollama"
    model_id: str = Field(min_length=1)
    review_status: Literal["AI_GENERATED_NOT_REVIEWED"] = "AI_GENERATED_NOT_REVIEWED"
    review_note: str = Field(min_length=1)
    limitations: list[str] = Field(
        default_factory=list,
        description="Identifiers defined in docs/LIMITATIONS.md that apply to this decision.",
    )
