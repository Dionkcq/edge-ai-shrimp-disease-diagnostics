"""Aspect-preserving resize-and-pad, replicating Ultralytics' ``LetterBox``.

This module owns the *only* coordinate transform in the system. Everything the
model reports is in the letterboxed input frame; everything a client sees is
normalized in the original, EXIF-corrected frame. If this is off by a pixel, every
box in the product is off by a pixel -- and the median white-spot target is about
eleven pixels at 640, so one pixel is a ten percent error.

The reproduction is exact and deliberate, including the parts that look like typos::

    r          = min(size / h, size / w)
    new_unpad  = (round(w * r), round(h * r))
    dw, dh     = (size - new_unpad[0]) / 2, (size - new_unpad[1]) / 2
    left, top  = round(dw - 0.1), round(dh - 0.1)

The ``-0.1`` breaks the banker's-rounding tie in ``round()`` toward the lower
value, so a half pixel of padding lands on the top-left rather than being split.
Using plain ``dw`` for the inverse mapping instead of ``left`` reintroduces up to
a one-pixel shift.

No mean/std normalization is applied: an Ultralytics export expects ``/255.0`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

#: Ultralytics' letterbox fill colour.
PAD_VALUE = 114


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """The forward mapping original -> letterboxed input, and its inverse.

    ``pad_left``/``pad_top`` are integers because they are the number of padded
    pixel columns/rows actually written, not the ideal half-difference.
    """

    original_width: int
    original_height: int
    input_size: int
    scale: float
    pad_left: int
    pad_top: int
    resized_width: int
    resized_height: int

    @classmethod
    def from_size(cls, width: int, height: int, input_size: int) -> LetterboxTransform:
        if width < 1 or height < 1:
            raise ValueError("image dimensions must be positive")
        if input_size < 1:
            raise ValueError("input size must be positive")
        scale = min(input_size / height, input_size / width)
        new_w = round(width * scale)
        new_h = round(height * scale)
        pad_w = (input_size - new_w) / 2
        pad_h = (input_size - new_h) / 2
        return cls(
            original_width=width,
            original_height=height,
            input_size=input_size,
            scale=scale,
            pad_left=round(pad_w - 0.1),
            pad_top=round(pad_h - 0.1),
            resized_width=new_w,
            resized_height=new_h,
        )

    def forward_xyxy(
        self, box: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        """Map an original-frame pixel box into the letterboxed input frame."""
        x1, y1, x2, y2 = box
        return (
            x1 * self.scale + self.pad_left,
            y1 * self.scale + self.pad_top,
            x2 * self.scale + self.pad_left,
            y2 * self.scale + self.pad_top,
        )

    def inverse_xyxy_pixels(self, boxes: np.ndarray) -> np.ndarray:
        """Map ``(n, 4)`` letterbox-frame boxes back to original-frame pixels."""
        out = np.asarray(boxes, dtype=np.float64).copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - self.pad_left) / self.scale
        out[:, [1, 3]] = (out[:, [1, 3]] - self.pad_top) / self.scale
        return out

    def inverse_xyxy_normalized(self, boxes: np.ndarray) -> np.ndarray:
        """Map ``(n, 4)`` letterbox-frame boxes to normalized original coordinates.

        Clipped to ``[0, 1]``: a detection whose box overlaps the grey padding would
        otherwise be reported as lying outside the photograph.
        """
        pixels = self.inverse_xyxy_pixels(boxes)
        pixels[:, [0, 2]] /= self.original_width
        pixels[:, [1, 3]] /= self.original_height
        clipped: np.ndarray = np.clip(pixels, 0.0, 1.0)
        return clipped


def letterbox_image(image: np.ndarray, input_size: int) -> tuple[np.ndarray, LetterboxTransform]:
    """Return a ``(1, 3, size, size)`` float32 NCHW batch and its transform.

    Input is ``(h, w, 3)`` ``uint8`` RGB. Output channel order stays RGB and the
    scale is ``/255.0``.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (h, w, 3) RGB array, got shape {image.shape}")
    height, width = int(image.shape[0]), int(image.shape[1])
    transform = LetterboxTransform.from_size(width, height, input_size)

    if (transform.resized_width, transform.resized_height) != (width, height):
        resized = np.asarray(
            Image.fromarray(image, mode="RGB").resize(
                (transform.resized_width, transform.resized_height),
                resample=Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
    else:
        resized = image

    canvas = np.full((input_size, input_size, 3), PAD_VALUE, dtype=np.uint8)
    top, left = transform.pad_top, transform.pad_left
    canvas[top : top + transform.resized_height, left : left + transform.resized_width] = resized

    batch = canvas.astype(np.float32) / 255.0
    return np.ascontiguousarray(batch.transpose(2, 0, 1)[None, ...]), transform
