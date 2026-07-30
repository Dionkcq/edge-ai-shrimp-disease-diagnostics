"""Decoding an untrusted upload into a normalized RGB array.

The order of the checks is the security design, cheapest and most decisive first:

1. non-empty and within the byte cap;
2. **magic bytes** -- the format is decided by the first eight bytes, never by the
   client's ``Content-Type`` or filename;
3. ``Image.open`` (header only) with the decoded format cross-checked against the
   sniff, so a PNG wearing a JPEG header is rejected rather than decoded;
4. header dimensions and pixel count, *before* ``load()`` allocates anything;
5. ``load()`` inside a guard that turns every decoder failure into one opaque 422.

Pillow's defaults are traps and each is overridden explicitly:

* ``MAX_IMAGE_PIXELS`` defaults to ~89.5 Mpx and between 1x and 2x that Pillow
  emits a *warning*, not an exception -- the stock configuration silently decodes
  up to ~179 Mpx. Both the limit and the warning-to-error promotion are set here.
* ``LOAD_TRUNCATED_IMAGES`` stays ``False``: a truncated JPEG must raise, not
  yield grey rows that the quality gate would then happily score.
* ``ImageOps.exif_transpose`` runs before anything measures the image, and the
  response reports post-transpose dimensions.

Nothing derived from the file -- filename, EXIF, GPS, ICC profile, camera make --
is returned or logged. ``source_mode`` is the single echoed fact, because an
inverted CMYK conversion is otherwise undebuggable.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from shrimp_screening.api.errors import ApiProblemError
from shrimp_screening.contracts.enums import ProblemCode
from shrimp_screening.contracts.screening import ImageInfo

#: Mirrors ``Settings.max_upload_bytes``; repeated here so a direct call to
#: ``decode_image`` in a test or a script is bounded too.
MAX_UPLOAD_BYTES = 12 * 1024 * 1024

#: 40 Mpx is comfortably above a 2048x2048 phone capture and far below the point
#: where a decode would matter for memory on a two-core laptop.
MAX_IMAGE_PIXELS = 40_000_000
MAX_DIMENSION = 8_000

Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
ImageFile.LOAD_TRUNCATED_IMAGES = False

_JPEG_MAGIC = b"\xff\xd8\xff"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

_ORIENTATION_TAG = 0x0112

#: 16-bit integer modes. ``convert("RGB")`` clips these to white instead of
#: rescaling, so they are rescaled explicitly from the full 16-bit range.
_SIXTEEN_BIT_MODES = frozenset({"I;16", "I;16B", "I;16L", "I;16N", "I"})


class IntakeError(ApiProblemError):
    """An upload that cannot be turned into an image safely.

    Subclasses :class:`ApiProblemError` so the API layer needs no translation step and
    cannot accidentally render a decoder message into a response body.
    """


def _reject(code: ProblemCode, detail: str, status: int) -> IntakeError:
    return IntakeError(code, status, detail)


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """An image that has passed intake.

    ``array`` is ``(height, width, 3)`` ``uint8`` RGB in the EXIF-corrected frame.
    """

    array: np.ndarray
    info: ImageInfo


def sniff_format(data: bytes) -> str:
    """Return ``"JPEG"`` or ``"PNG"`` from the leading magic bytes, or reject."""
    if data.startswith(_JPEG_MAGIC):
        return "JPEG"
    if data.startswith(_PNG_MAGIC):
        return "PNG"
    raise _reject(
        ProblemCode.UNSUPPORTED_MEDIA_TYPE,
        "Only JPEG and PNG photographs are accepted.",
        415,
    )


def _exif_orientation_transposes(image: Image.Image) -> bool:
    """True when the file's EXIF orientation is one that changes the pixels.

    Comparing sizes before and after is not sufficient: orientation 3 is a 180
    degree rotation, which preserves the size while still changing the pixels, and
    orientations 2 and 4 are mirrors. All of them must be reported as transposed.
    """
    try:
        exif = image.getexif()
    except (AttributeError, OSError, ValueError, SyntaxError):
        return False
    orientation = exif.get(_ORIENTATION_TAG)
    return isinstance(orientation, int) and 2 <= orientation <= 8


def _to_rgb(image: Image.Image) -> Image.Image:
    """Normalize any supported Pillow mode to 8-bit RGB, deterministically."""
    if image.mode in _SIXTEEN_BIT_MODES:
        # Fixed 16-bit -> 8-bit scaling. An adaptive per-image stretch would make
        # the pixel values depend on the brightest pixel in the frame, so the same
        # shrimp photographed twice would be scored on two different scales.
        pixels = np.asarray(image, dtype=np.float64)
        scaled = np.clip(pixels * (255.0 / 65535.0), 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(scaled, mode="L").convert("RGB")
    if image.mode in {"RGBA", "LA", "PA", "P"}:
        # Flatten onto a defined background rather than letting Pillow drop alpha,
        # so a transparent PNG cannot smuggle in undefined pixel values.
        rgba = image.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        canvas.alpha_composite(rgba)
        return canvas.convert("RGB")
    return image.convert("RGB")


def _check_pixel_bounds(width: int, height: int) -> None:
    if (
        width < 1
        or height < 1
        or max(width, height) > MAX_DIMENSION
        or width * height > MAX_IMAGE_PIXELS
    ):
        raise _reject(
            ProblemCode.PAYLOAD_TOO_LARGE,
            "The photograph's pixel dimensions exceed the safety limit.",
            413,
        )


def decode_image(data: bytes, *, max_bytes: int = MAX_UPLOAD_BYTES) -> DecodedImage:
    """Decode an untrusted JPEG or PNG into a normalized RGB array."""
    if not data:
        raise _reject(ProblemCode.UNDECODABLE_IMAGE, "The uploaded photograph is empty.", 422)
    if len(data) > max_bytes:
        raise _reject(
            ProblemCode.PAYLOAD_TOO_LARGE,
            f"The upload exceeds the {max_bytes // (1024 * 1024)} MB limit.",
            413,
        )
    expected = sniff_format(data)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as opened:
                # MPO is multi-picture JPEG: what phone cameras actually produce.
                # Accept it as JPEG and use the first frame only.
                declared = opened.format
                actual = "JPEG" if declared == "MPO" else declared
                if actual != expected:
                    raise _reject(
                        ProblemCode.UNDECODABLE_IMAGE,
                        "The file header and its decoded format disagree.",
                        422,
                    )
                if declared != "MPO" and getattr(opened, "n_frames", 1) > 1:
                    raise _reject(
                        ProblemCode.UNDECODABLE_IMAGE,
                        "Animated images are not accepted; send a single photograph.",
                        422,
                    )
                if declared == "MPO":
                    opened.seek(0)

                _check_pixel_bounds(*opened.size)
                source_mode = opened.mode
                # A large or malformed ICC profile is a memory vector and we do no
                # colour management, so drop it before load() can act on it.
                opened.info.pop("icc_profile", None)
                exif_transposed = _exif_orientation_transposes(opened)
                opened.load()
                # Re-check after load(): not every plugin routes through Pillow's
                # internal bomb check, and some report a placeholder size first.
                _check_pixel_bounds(*opened.size)
                upright = ImageOps.exif_transpose(opened) or opened
                array = np.asarray(_to_rgb(upright), dtype=np.uint8).copy()
    except IntakeError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        # Pillow raises this from inside `Image.open()` once the declared pixel count
        # passes 2x MAX_IMAGE_PIXELS, i.e. before our own bounds check can run. It is
        # the same condition, so it must produce the same answer: "too large", not
        # "undecodable". Otherwise a client cannot tell an oversized photograph from
        # a corrupt file and has no idea that resizing would fix it.
        raise _reject(
            ProblemCode.PAYLOAD_TOO_LARGE,
            "The photograph's pixel dimensions exceed the safety limit.",
            413,
        ) from exc
    except (
        UnidentifiedImageError,
        OSError,
        SyntaxError,
        ValueError,
        EOFError,
    ) as exc:
        # The decoder's own message can quote file content; it never leaves here.
        raise _reject(
            ProblemCode.UNDECODABLE_IMAGE,
            "The uploaded file could not be decoded safely.",
            422,
        ) from exc

    if array.ndim != 3 or array.shape[2] != 3:
        raise _reject(
            ProblemCode.UNDECODABLE_IMAGE,
            "The uploaded file could not be decoded safely.",
            422,
        )
    return DecodedImage(
        array=array,
        info=ImageInfo(
            width=int(array.shape[1]),
            height=int(array.shape[0]),
            source_mode=source_mode,
            exif_transposed=exif_transposed,
        ),
    )
