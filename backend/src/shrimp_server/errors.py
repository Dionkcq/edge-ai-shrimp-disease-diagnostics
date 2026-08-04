"""Rendering :class:`ApiProblemError` as ``application/problem+json`` (RFC 9457).

The error type itself lives in :mod:`shrimp_screening.problems`, because domain
code raises it. This module is only the HTTP half: status line, ``Retry-After``,
media type, and the two handlers FastAPI is given.

One rule governs everything here: **nothing derived from the request body, its
filename, its headers or its EXIF may reach a response or a log line.** Detail
strings are constants chosen at the raise site, never interpolated from input.
The only per-request value that escapes is the server-generated ULID.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from shrimp_screening.contracts.enums import ProblemCode
from shrimp_screening.problems import ApiProblemError

PROBLEM_MEDIA_TYPE = "application/problem+json"


def problem_response(problem: ApiProblemError, request_id: str | None = None) -> JSONResponse:
    """Render an :class:`ApiProblemError` as a problem+json response."""
    headers: dict[str, str] = {}
    if problem.retry_after_seconds is not None:
        headers["Retry-After"] = str(problem.retry_after_seconds)
    return JSONResponse(
        problem.to_document(request_id).model_dump(mode="json"),
        status_code=problem.status,
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


async def api_problem_handler(request: Request, exc: Exception) -> JSONResponse:
    """Exception handler registered for :class:`ApiProblemError`."""
    assert isinstance(exc, ApiProblemError)
    return problem_response(exc, getattr(request.state, "request_id", None))


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler.

    The exception text is deliberately discarded: a traceback string can contain a
    filesystem path, a decoded byte range or a Pillow message quoting file content.
    Correlate with the server log using ``request_id`` instead.
    """
    del exc
    return problem_response(
        ApiProblemError(
            ProblemCode.INTERNAL_ERROR,
            500,
            "The request could not be completed. No image was retained.",
        ),
        getattr(request.state, "request_id", None),
    )
