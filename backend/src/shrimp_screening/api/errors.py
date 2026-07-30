"""``application/problem+json`` responses (RFC 9457).

One rule governs everything in this module: **nothing derived from the request
body, its filename, its headers or its EXIF may reach a response or a log line.**
Detail strings are constants chosen at the raise site, never interpolated from
input. The only per-request value that escapes is the server-generated ULID.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse

from shrimp_screening.contracts.enums import ProblemCode
from shrimp_screening.contracts.problem import ProblemDetail, problem_type_for

PROBLEM_MEDIA_TYPE = "application/problem+json"

#: Human-readable titles, fixed per code so they are translatable and never leak input.
_TITLES: dict[ProblemCode, str] = {
    ProblemCode.MALFORMED_REQUEST: "Malformed request",
    ProblemCode.PAYLOAD_TOO_LARGE: "Payload too large",
    ProblemCode.UNSUPPORTED_MEDIA_TYPE: "Unsupported media type",
    ProblemCode.UNDECODABLE_IMAGE: "Undecodable image",
    ProblemCode.SERVICE_BUSY: "Service busy",
    ProblemCode.NOT_FOUND: "Not found",
    ProblemCode.INTERNAL_ERROR: "Internal error",
}


class ApiProblemError(Exception):
    """An error that is safe to render to a client verbatim."""

    def __init__(
        self,
        code: ProblemCode,
        status: int,
        detail: str,
        *,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(detail)
        self.code = code
        self.status = status
        self.detail = detail
        self.retry_after_seconds = retry_after_seconds

    def to_document(self, request_id: str | None) -> ProblemDetail:
        return ProblemDetail(
            type=problem_type_for(self.code),
            title=_TITLES[self.code],
            status=self.status,
            detail=self.detail,
            code=self.code,
            request_id=request_id,
            retry_after_seconds=self.retry_after_seconds,
        )


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
