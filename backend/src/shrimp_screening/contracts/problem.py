"""RFC 9457 ``application/problem+json`` error bodies.

`code` is the stable machine-readable field; `type`, `title` and `detail` are for
humans and may be reworded without a contract break. `detail` never echoes anything
derived from the request body, filename or EXIF.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from shrimp_screening.contracts.enums import ProblemCode

#: Registered-relative URIs. Kept relative so the app has no external base URL and
#: therefore no way to leak a hostname into an offline deployment's error body.
PROBLEM_TYPE_BASE = "/problems/"


class ProblemDetail(BaseModel):
    """RFC 9457 problem document with a stable ``code`` extension member."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: str = Field(description="Problem type URI reference, relative to the app origin.")
    title: str
    status: int = Field(ge=400, le=599)
    detail: str
    code: ProblemCode
    request_id: str | None = Field(
        default=None,
        description="ULID of the request, for correlating with server logs.",
    )
    retry_after_seconds: int | None = Field(
        default=None,
        ge=0,
        description="Present for SERVICE_BUSY; mirrors the Retry-After header.",
    )


def problem_type_for(code: ProblemCode) -> str:
    return f"{PROBLEM_TYPE_BASE}{code.value.lower().replace('_', '-')}"
