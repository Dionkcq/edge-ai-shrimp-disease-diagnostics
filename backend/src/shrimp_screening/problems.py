"""The error type that is safe to render to a client.

This lives in the domain rather than in the server because the code that *raises*
it is domain code: the image intake path and the multipart reader both reject
untrusted input long before anything HTTP-shaped is involved. Putting the class
here is what lets ``shrimp_server`` depend on ``shrimp_screening`` and never the
other way round.

Rendering it -- status line, headers, ``application/problem+json`` body -- is the
server's job and lives in :mod:`shrimp_server.errors`.

One rule governs the content: **nothing derived from the request body, its
filename, its headers or its EXIF may reach a response or a log line.** Detail
strings are constants chosen at the raise site, never interpolated from input.
"""

from __future__ import annotations

from shrimp_screening.contracts.enums import ProblemCode
from shrimp_screening.contracts.problem import ProblemDetail, problem_type_for

#: Human-readable titles, fixed per code so they are translatable and never leak input.
_TITLES: dict[ProblemCode, str] = {
    ProblemCode.MALFORMED_REQUEST: "Malformed request",
    ProblemCode.PAYLOAD_TOO_LARGE: "Payload too large",
    ProblemCode.UNSUPPORTED_MEDIA_TYPE: "Unsupported media type",
    ProblemCode.UNDECODABLE_IMAGE: "Undecodable image",
    ProblemCode.SERVICE_BUSY: "Service busy",
    ProblemCode.NOT_FOUND: "Not found",
    ProblemCode.INTERNAL_ERROR: "Internal error",
    ProblemCode.ADVICE_UNAVAILABLE: "Advice unavailable",
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
