"""Hostile and awkward uploads.

Every case here is either an attack (decompression bomb, format confusion, an
oversized body) or a real-world format that a naive implementation silently gets
wrong (MPO from a phone camera, 16-bit PNG, CMYK JPEG, EXIF orientation).

The distinction matters for the ones that look pedantic. A 16-bit PNG passed
through `convert("RGB")` clips to pure white rather than rescaling, so the quality
gate would score a blank frame and the detector would run on nothing. That is not a
hypothetical; it is Pillow's documented behaviour.
"""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import pytest
from PIL import Image

from shrimp_screening.contracts.enums import ProblemCode
from shrimp_screening.imaging.intake import (
    IntakeError,
    decode_image,
    sniff_format,
)
from shrimp_screening.settings import MAX_DIMENSION
from tests.support.factories import client_for, make_image_bytes

# ---------------------------------------------------------------------------
# Helpers that build hostile inputs.
# ---------------------------------------------------------------------------


def _png_with_declared_size(width: int, height: int) -> bytes:
    """A structurally valid PNG whose IHDR *claims* an enormous size.

    Built by rewriting the IHDR of a small real PNG and repairing its CRC, so the
    header can be parsed without ever allocating the claimed pixels. This is the
    cheap form of a decompression bomb and the one the dimension check must catch
    before `load()`.
    """
    original = make_image_bytes(width=64, height=64, image_format="PNG")
    signature, rest = original[:8], original[8:]
    length = struct.unpack(">I", rest[:4])[0]
    assert rest[4:8] == b"IHDR"
    body = bytearray(rest[8 : 8 + length])
    body[0:4] = struct.pack(">I", width)
    body[4:8] = struct.pack(">I", height)
    crc = zlib.crc32(b"IHDR" + bytes(body)) & 0xFFFFFFFF
    return (
        signature
        + struct.pack(">I", length)
        + b"IHDR"
        + bytes(body)
        + struct.pack(">I", crc)
        + rest[8 + length + 4 :]
    )


def _apng_bytes() -> bytes:
    frames = [Image.new("RGB", (700, 700), (c, c, c)) for c in (40, 120)]
    buffer = io.BytesIO()
    frames[0].save(buffer, format="PNG", save_all=True, append_images=frames[1:])
    return buffer.getvalue()


def _jpeg_with_exif(**tags: object) -> bytes:
    """A JPEG carrying chosen EXIF tags, used to prove they are never echoed."""
    exif = Image.Exif()
    for name, value in tags.items():
        exif[{"orientation": 0x0112, "make": 0x010F, "model": 0x0110}[name]] = value
    image = Image.fromarray(
        np.random.default_rng(3).integers(0, 255, (768, 1024, 3), dtype=np.uint8)
    )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=90, exif=exif)
    return buffer.getvalue()


def _problem(exc_info: pytest.ExceptionInfo[IntakeError]) -> tuple[ProblemCode, int]:
    problem = exc_info.value
    return problem.code, problem.status


# ---------------------------------------------------------------------------
# Format is decided by content, never by what the client claims.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"\x00" * 64, id="nulls"),
        pytest.param(b"GIF89a" + b"\x00" * 32, id="gif"),
        pytest.param(b"BM" + b"\x00" * 32, id="bmp"),
        pytest.param(b"<svg xmlns='http://www.w3.org/2000/svg'></svg>", id="svg"),
        pytest.param(b"%PDF-1.7\n", id="pdf"),
        pytest.param(b"RIFF\x00\x00\x00\x00WEBPVP8 ", id="webp"),
        pytest.param(b"II*\x00" + b"\x00" * 32, id="tiff"),
        pytest.param(b"#!/bin/sh\nrm -rf /\n", id="shell-script"),
    ],
)
def test_only_jpeg_and_png_are_accepted(payload: bytes) -> None:
    with pytest.raises(IntakeError) as exc_info:
        decode_image(payload)
    code, status = _problem(exc_info)
    # An empty body is a decode failure; everything else is an unsupported type.
    assert (code, status) in {
        (ProblemCode.UNSUPPORTED_MEDIA_TYPE, 415),
        (ProblemCode.UNDECODABLE_IMAGE, 422),
    }


def test_sniff_ignores_a_lying_content_type() -> None:
    """`sniff_format` reads bytes. It has no access to a declared type at all."""
    assert sniff_format(make_image_bytes(image_format="PNG")) == "PNG"
    assert sniff_format(make_image_bytes(image_format="JPEG")) == "JPEG"


def test_a_png_wearing_a_jpeg_header_is_rejected_not_decoded() -> None:
    """Format confusion: the sniff and the decoder must be cross-checked."""
    png = make_image_bytes(image_format="PNG")
    disguised = b"\xff\xd8\xff" + png[3:]
    with pytest.raises(IntakeError) as exc_info:
        decode_image(disguised)
    assert _problem(exc_info)[1] == 422


def test_a_truncated_jpeg_raises_instead_of_yielding_grey_rows() -> None:
    """`LOAD_TRUNCATED_IMAGES` must stay False, or half an image gets scored."""
    full = make_image_bytes(width=1024, height=768, image_format="JPEG", quality=92)
    with pytest.raises(IntakeError) as exc_info:
        decode_image(full[: len(full) // 2])
    assert _problem(exc_info) == (ProblemCode.UNDECODABLE_IMAGE, 422)


def test_a_truncated_png_is_rejected() -> None:
    """The interrupted-upload case: a valid header with the pixel data cut short."""
    png = make_image_bytes(width=700, height=700, image_format="PNG")
    with pytest.raises(IntakeError) as exc_info:
        decode_image(png[: int(len(png) * 0.6)])
    assert _problem(exc_info) == (ProblemCode.UNDECODABLE_IMAGE, 422)


# ---------------------------------------------------------------------------
# Size and pixel-count limits, enforced before any allocation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("width", "height"),
    [
        (MAX_DIMENSION + 1, 100),
        (100, MAX_DIMENSION + 1),
        (30_000, 30_000),
        (0xFFFF, 0xFFFF),
    ],
)
def test_oversized_declared_dimensions_are_refused_before_decoding(width: int, height: int) -> None:
    with pytest.raises(IntakeError) as exc_info:
        decode_image(_png_with_declared_size(width, height))
    assert _problem(exc_info) == (ProblemCode.PAYLOAD_TOO_LARGE, 413)


def test_a_body_over_the_byte_cap_is_refused() -> None:
    with pytest.raises(IntakeError) as exc_info:
        decode_image(make_image_bytes(width=700, height=700), max_bytes=1024)
    assert _problem(exc_info) == (ProblemCode.PAYLOAD_TOO_LARGE, 413)


def test_an_animated_png_is_refused() -> None:
    with pytest.raises(IntakeError) as exc_info:
        decode_image(_apng_bytes())
    assert _problem(exc_info) == (ProblemCode.UNDECODABLE_IMAGE, 422)


# ---------------------------------------------------------------------------
# Formats that must work, because real cameras produce them.
# ---------------------------------------------------------------------------


def test_a_cmyk_jpeg_converts_to_rgb_and_echoes_its_source_mode() -> None:
    """`source_mode` is echoed so an inverted Adobe CMYK result is debuggable."""
    rgb = Image.fromarray(np.random.default_rng(1).integers(0, 255, (768, 1024, 3), dtype=np.uint8))
    buffer = io.BytesIO()
    rgb.convert("CMYK").save(buffer, format="JPEG", quality=92)
    decoded = decode_image(buffer.getvalue())
    assert decoded.array.shape == (768, 1024, 3)
    assert decoded.info.source_mode == "CMYK"


def test_a_16_bit_png_is_rescaled_rather_than_clipped_to_white() -> None:
    """`convert("RGB")` on `I;16` clips to white; the result must not be blank."""
    values = np.linspace(0, 65535, 1024 * 768, dtype=np.uint16).reshape(768, 1024)
    buffer = io.BytesIO()
    # No `mode=` argument: Pillow 12 deprecates using it to reinterpret dtypes, and
    # a uint16 array already maps to I;16.
    Image.fromarray(values).save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue())
    assert decoded.info.source_mode in {"I;16", "I"}
    # A clipping bug produces an all-255 array; a correct rescale spans the range.
    assert decoded.array.min() < 10
    assert decoded.array.max() > 245
    assert 100 < float(decoded.array.mean()) < 155


def test_a_transparent_png_is_flattened_onto_a_defined_background() -> None:
    """Alpha must not leave undefined pixel values behind."""
    image = Image.new("RGBA", (1024, 768), (0, 0, 0, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue())
    assert decoded.info.source_mode == "RGBA"
    assert decoded.array.shape == (768, 1024, 3)
    assert (decoded.array == 255).all(), "fully transparent must flatten to the background"


def test_a_palette_png_decodes_to_three_channels() -> None:
    decoded = decode_image(make_image_bytes(width=800, height=700, image_format="PNG", mode="P"))
    assert decoded.array.shape == (700, 800, 3)
    assert decoded.info.source_mode == "P"


def test_a_greyscale_jpeg_decodes_to_three_channels() -> None:
    decoded = decode_image(
        make_image_bytes(width=800, height=700, image_format="JPEG", mode="L", quality=92)
    )
    assert decoded.array.shape == (700, 800, 3)


# ---------------------------------------------------------------------------
# EXIF: applied to the pixels, never echoed to the client.
# ---------------------------------------------------------------------------


def test_exif_orientation_is_applied_and_reported() -> None:
    """Orientation 6 rotates 90 degrees, so the reported size must be swapped."""
    decoded = decode_image(_jpeg_with_exif(orientation=6))
    assert decoded.info.exif_transposed is True
    assert (decoded.info.width, decoded.info.height) == (768, 1024)
    assert decoded.array.shape == (1024, 768, 3)


def test_orientation_3_is_reported_even_though_the_size_is_unchanged() -> None:
    """A 180-degree rotation preserves size; comparing sizes would miss it."""
    decoded = decode_image(_jpeg_with_exif(orientation=3))
    assert decoded.info.exif_transposed is True
    assert (decoded.info.width, decoded.info.height) == (1024, 768)


def test_an_image_without_exif_is_not_reported_as_transposed() -> None:
    decoded = decode_image(make_image_bytes(width=1024, height=768, quality=92))
    assert decoded.info.exif_transposed is False


def test_camera_identity_from_exif_never_reaches_the_response() -> None:
    """EXIF on a farm photograph is sensitive; none of it may be echoed."""
    payload = _jpeg_with_exif(make="SECRETCAMERAMAKE", model="SECRETCAMERAMODEL")
    with client_for("unavailable") as client:
        response = client.post(
            "/api/v1/screenings", files={"image": ("x.jpg", payload, "image/jpeg")}
        )
    assert response.status_code == 200
    assert "SECRETCAMERA" not in response.text
    # And the structured image block carries only the four declared fields.
    assert set(response.json()["image"]) == {
        "width",
        "height",
        "source_mode",
        "exif_transposed",
    }


def test_the_decoded_array_is_a_private_copy() -> None:
    """The array must not be a view onto the decoder's internal buffer."""
    decoded = decode_image(make_image_bytes(width=700, height=700, quality=92))
    assert decoded.array.flags.owndata
    assert decoded.array.dtype == np.uint8


# ---------------------------------------------------------------------------
# The HTTP body cap, enforced while reading rather than from Content-Length.
# ---------------------------------------------------------------------------


def test_an_oversized_upload_is_refused_with_413() -> None:
    small_cap = 64 * 1024
    payload = make_image_bytes(width=1400, height=1400, image_format="PNG")
    assert len(payload) > small_cap
    with client_for("unavailable", max_upload_bytes=small_cap) as client:
        response = client.post(
            "/api/v1/screenings", files={"image": ("big.png", payload, "image/png")}
        )
    assert response.status_code == 413
    assert response.json()["code"] == "PAYLOAD_TOO_LARGE"


def test_a_lying_content_length_cannot_smuggle_a_large_body() -> None:
    """The counter is on bytes actually read, so a false header changes nothing."""
    small_cap = 32 * 1024
    payload = make_image_bytes(width=1400, height=1400, image_format="PNG")
    with client_for("unavailable", max_upload_bytes=small_cap) as client:
        response = client.post(
            "/api/v1/screenings",
            files={"image": ("big.png", payload, "image/png")},
            headers={"content-length": "10"},
        )
    assert response.status_code in {400, 413}


def test_too_many_form_parts_is_refused() -> None:
    with client_for("unavailable") as client:
        response = client.post(
            "/api/v1/screenings",
            files=[(f"f{i}", (f"{i}.txt", b"x", "text/plain")) for i in range(12)],
        )
    assert response.status_code == 400
    assert response.json()["code"] == "MALFORMED_REQUEST"
