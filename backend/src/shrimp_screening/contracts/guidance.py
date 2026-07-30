"""The response contract for guidance retrieval.

Every guidance response carries its citations and its review status inline. A
client cannot render the advice while omitting the fact that it has not been
reviewed by a professional, because there is no field arrangement in which the
advice arrives without it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from shrimp_screening.contracts.enums import Decision

_STRICT = ConfigDict(extra="forbid", frozen=True)


class CitedSource(BaseModel):
    """One reference behind a guidance item."""

    model_config = _STRICT

    id: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    url: str = Field(min_length=1)
    accessed_on: str = Field(min_length=1, max_length=32)


class GuidanceDocument(BaseModel):
    """Response body of ``GET /api/v1/guidance/{decision}``."""

    model_config = _STRICT

    decision: Decision
    id: str = Field(min_length=1, max_length=64)
    headline: str = Field(min_length=1)
    body: str = Field(min_length=1)
    sources: list[CitedSource] = Field(min_length=1)
    review_status: str = Field(min_length=1)
    review_note: str = Field(min_length=1)
    limitations: list[str] = Field(
        default_factory=list,
        description="Identifiers defined in docs/LIMITATIONS.md that apply to this decision.",
    )
