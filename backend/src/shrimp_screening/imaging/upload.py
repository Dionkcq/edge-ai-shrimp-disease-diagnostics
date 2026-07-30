"""Bounded, memory-only multipart reading.

Why this exists instead of ``fastapi.UploadFile``
-------------------------------------------------
Starlette's ``UploadFile`` is backed by a ``SpooledTemporaryFile`` that rolls over
to **disk** past 1 MiB, and neither Starlette nor uvicorn caps the body by
default. A 2 GB POST therefore writes 2 GB into ``/tmp`` before the handler is
ever entered. Two of this project's stated properties -- images are ephemeral and
never touch disk, and the body is capped *while being read* -- cannot be honoured
on top of that machinery.

So the multipart stream is parsed directly with ``python_multipart``'s streaming
parser, into a bounded ``bytearray``:

* the running byte count is checked on every chunk, so ``Content-Length`` is never
  trusted and an oversized upload is abandoned mid-flight;
* nothing is written to a file object at any point;
* the client-supplied filename and per-part ``Content-Type`` are read only to be
  ignored -- the format is decided by magic bytes in ``intake.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.requests import Request

from shrimp_screening.api.errors import ApiProblemError
from shrimp_screening.contracts.enums import ProblemCode

#: A screening request carries exactly one file. Anything else is a client bug or
#: an attempt to make the parser do work proportional to attacker-controlled input.
MAX_PARTS = 8

#: Bound on the ``name="..."`` bookkeeping we are willing to buffer per part.
_MAX_HEADER_BYTES = 4096


def _too_large(limit: int) -> ApiProblemError:
    return ApiProblemError(
        ProblemCode.PAYLOAD_TOO_LARGE,
        413,
        f"The upload exceeds the {limit // (1024 * 1024)} MB limit.",
    )


def _malformed() -> ApiProblemError:
    return ApiProblemError(
        ProblemCode.MALFORMED_REQUEST,
        400,
        "The multipart request is malformed.",
    )


@dataclass(slots=True)
class _FieldCollector:
    """Accumulates only the bytes of the one part we care about."""

    field_name: str
    limit: int
    data: bytearray = field(default_factory=bytearray)
    found: bool = False
    completed_image_parts: int = 0
    _parts_seen: int = 0
    _header_name: bytearray = field(default_factory=bytearray)
    _header_value: bytearray = field(default_factory=bytearray)
    _capturing: bool = False

    def on_part_begin(self) -> None:
        self._parts_seen += 1
        if self._parts_seen > MAX_PARTS:
            raise ApiProblemError(
                ProblemCode.MALFORMED_REQUEST, 400, "The request has too many form parts."
            )
        self._capturing = False
        self._header_name.clear()
        self._header_value.clear()

    def on_header_field(self, data: bytes, start: int, end: int) -> None:
        if len(self._header_name) + (end - start) <= _MAX_HEADER_BYTES:
            self._header_name.extend(data[start:end])

    def on_header_value(self, data: bytes, start: int, end: int) -> None:
        if len(self._header_value) + (end - start) <= _MAX_HEADER_BYTES:
            self._header_value.extend(data[start:end])

    def on_header_end(self) -> None:
        name = bytes(self._header_name).lower()
        value = bytes(self._header_value)
        self._header_name.clear()
        self._header_value.clear()
        if name != b"content-disposition":
            return
        _, options = parse_options_header(value)
        part_name = options.get(b"name", b"")
        if part_name.decode("latin-1") == self.field_name:
            if self.found:
                raise _malformed()
            self._capturing = True
            self.found = True

    def on_part_data(self, data: bytes, start: int, end: int) -> None:
        if not self._capturing:
            return
        chunk = data[start:end]
        if len(self.data) + len(chunk) > self.limit:
            raise _too_large(self.limit)
        self.data.extend(chunk)

    def on_part_end(self) -> None:
        if self._capturing:
            self.completed_image_parts += 1
        self._capturing = False


async def read_single_image_part(request: Request, field_name: str, limit: int) -> bytes:
    """Return the raw bytes of one multipart file field, never exceeding ``limit``.

    Raises :class:`ApiProblemError` for an unusable content type, a missing field or an
    oversized body. The response body is consumed at most once.
    """
    content_type_header = request.headers.get("content-type", "")
    media_type, options = parse_options_header(content_type_header.encode("latin-1"))
    if media_type.lower() != b"multipart/form-data":
        raise ApiProblemError(
            ProblemCode.UNSUPPORTED_MEDIA_TYPE,
            415,
            "Send the photograph as multipart/form-data.",
        )
    boundary = options.get(b"boundary")
    if not boundary:
        raise ApiProblemError(
            ProblemCode.MALFORMED_REQUEST,
            400,
            "The multipart body has no boundary.",
        )

    collector = _FieldCollector(field_name=field_name, limit=limit)
    parser = MultipartParser(
        boundary,
        callbacks={
            "on_part_begin": collector.on_part_begin,
            "on_header_field": collector.on_header_field,
            "on_header_value": collector.on_header_value,
            "on_header_end": collector.on_header_end,
            "on_part_data": collector.on_part_data,
            "on_part_end": collector.on_part_end,
        },
    )

    received = 0
    try:
        async for chunk in request.stream():
            received += len(chunk)
            if received > limit:
                raise _too_large(limit)
            if chunk:
                parser.write(chunk)
        parser.finalize()
    except MultipartParseError as exc:
        raise _malformed() from exc

    if not collector.found:
        raise ApiProblemError(
            ProblemCode.MALFORMED_REQUEST,
            400,
            f"The request has no {field_name!r} file field.",
        )
    if collector.completed_image_parts != 1:
        raise _malformed()
    return bytes(collector.data)
