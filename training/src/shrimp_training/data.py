# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dataset loading for the from-scratch trainer: images + YOLO-format labels.

Reads directly from a prepared dataset's ``manifest.json`` -- nothing produces or
consumes an Ultralytics-style ``dataset.yaml`` descriptor anymore. Preprocessing
duplicates the runtime's resize-and-pad ("letterbox") algorithm rather than
importing it: the backend and training packages are separately locked MIT/AGPL
trees that deliberately do not share code across the boundary (see
``acceptance.py``'s duplication of ``pipeline/gate.py``'s schema for the
established precedent, and ADR-0001).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from PIL import Image
from torch.utils.data import Dataset

from shrimp_training.dataset import PreparedDataset

PAD_VALUE = 114
_PARTITIONS = ("train", "validation", "test")


class DatasetLoadError(RuntimeError):
    """A prepared image or label could not be loaded for training."""


@dataclass(frozen=True, slots=True)
class _ImageRecord:
    image_path: Path
    label_path: Path


@dataclass(frozen=True, slots=True)
class _LetterboxParams:
    scale: float
    new_width: int
    new_height: int
    pad_left: int
    pad_top: int

    @classmethod
    def compute(cls, width: int, height: int, size: int) -> _LetterboxParams:
        scale = min(size / height, size / width)
        new_w, new_h = round(width * scale), round(height * scale)
        pad_left = round((size - new_w) / 2 - 0.1)
        pad_top = round((size - new_h) / 2 - 0.1)
        return cls(scale, new_w, new_h, pad_left, pad_top)


def _letterbox_image(image: NDArray[Any], size: int, params: _LetterboxParams) -> NDArray[Any]:
    if (params.new_width, params.new_height) != (image.shape[1], image.shape[0]):
        resized = np.asarray(
            Image.fromarray(image).resize(
                (params.new_width, params.new_height), resample=Image.Resampling.BILINEAR
            ),
            dtype=np.uint8,
        )
    else:
        resized = image
    canvas = np.full((size, size, 3), PAD_VALUE, dtype=np.uint8)
    canvas[
        params.pad_top : params.pad_top + params.new_height,
        params.pad_left : params.pad_left + params.new_width,
    ] = resized
    return canvas


def _load_labels(path: Path) -> NDArray[Any]:
    """Return ``(n, 5)`` rows of ``[class, cx, cy, w, h]``, all normalized [0, 1]."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetLoadError(f"cannot read label file: {path}") from exc
    rows: list[list[float]] = []
    for line in lines:
        fields = line.split()
        if len(fields) != 5:
            continue
        rows.append([float(value) for value in fields])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _rgb_to_hsv(rgb: NDArray[Any]) -> NDArray[Any]:
    """Vectorized RGB -> HSV, all channels in ``[0, 1]``. Standard colorsys algorithm.

    Implemented directly in numpy rather than round-tripped through Pillow's "HSV"
    image mode: Pillow's ``Image.fromarray(..., mode=...)`` is deprecated, and the
    mode can't simply be dropped here the way it can for a same-semantics RGB
    array -- an array of HSV values tagged "RGB" would be silently misread.
    """
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    value = rgb.max(axis=-1)
    minimum = rgb.min(axis=-1)
    delta = value - minimum
    safe_delta = np.where(delta > 0, delta, 1.0)

    saturation = np.where(value > 0, delta / np.where(value > 0, value, 1.0), 0.0)
    red_c = (value - red) / safe_delta
    green_c = (value - green) / safe_delta
    blue_c = (value - blue) / safe_delta

    hue = np.zeros_like(value)
    hue = np.where(value == blue, 4.0 + green_c - red_c, hue)
    hue = np.where(value == green, 2.0 + red_c - blue_c, hue)
    hue = np.where(value == red, blue_c - green_c, hue)
    hue = np.where(delta > 0, np.mod(hue / 6.0, 1.0), 0.0)
    return np.stack([hue, saturation, value], axis=-1)


def _hsv_to_rgb(hsv: NDArray[Any]) -> NDArray[Any]:
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    sector = np.floor(hue * 6.0)
    fraction = hue * 6.0 - sector
    sector_index = np.mod(sector.astype(np.int64), 6)

    p = value * (1.0 - saturation)
    q = value * (1.0 - saturation * fraction)
    t = value * (1.0 - saturation * (1.0 - fraction))

    conditions = [sector_index == index for index in range(6)]
    red = np.select(conditions, [value, q, p, p, t, value], default=value)
    green = np.select(conditions, [t, value, value, q, p, p], default=t)
    blue = np.select(conditions, [p, p, t, value, value, q], default=p)
    return np.stack([red, green, blue], axis=-1)


def _hsv_jitter(
    image: NDArray[Any],
    rng: random.Random,
    *,
    hue_gain: float = 0.015,
    saturation_gain: float = 0.7,
    value_gain: float = 0.4,
) -> NDArray[Any]:
    hsv = _rgb_to_hsv(image.astype(np.float64) / 255.0)
    hue_shift = rng.uniform(-1.0, 1.0) * hue_gain
    saturation_scale = 1.0 + rng.uniform(-1.0, 1.0) * saturation_gain
    value_scale = 1.0 + rng.uniform(-1.0, 1.0) * value_gain
    hsv[..., 0] = np.mod(hsv[..., 0] + hue_shift, 1.0)
    hsv[..., 1] = np.clip(hsv[..., 1] * saturation_scale, 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2] * value_scale, 0.0, 1.0)
    rgb = np.clip(_hsv_to_rgb(hsv) * 255.0, 0, 255)
    return rgb.astype(np.uint8)


class ShrimpDetectionDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """One prepared-dataset partition, yielding letterboxed images and canvas-space targets.

    Each target row is ``[class, cx, cy, w, h]`` in **absolute 640-canvas pixels**
    (not normalized), matching what ``loss.py``'s grid/anchor assignment expects.
    Train-only augmentation (horizontal flip + HSV jitter) is applied before
    letterboxing, so a flip only needs ``cx' = 1 - cx`` in original-image normalized
    space -- correct regardless of that image's own letterbox padding.
    """

    def __init__(
        self,
        prepared: PreparedDataset,
        partition: str,
        *,
        image_size: int = 640,
        augment: bool = False,
        seed: int = 20260730,
    ) -> None:
        if partition not in _PARTITIONS:
            raise ValueError(f"unknown partition: {partition!r}")
        try:
            document = json.loads(prepared.manifest.read_text(encoding="utf-8"))
        except OSError as exc:
            raise DatasetLoadError(f"cannot read prepared manifest: {prepared.manifest}") from exc
        root = prepared.manifest.parent
        self._records = [
            _ImageRecord(root / str(record["image_output"]), root / str(record["label_output"]))
            for record in document.get("images", [])
            if record.get("partition") == partition
        ]
        if not self._records:
            raise DatasetLoadError(f"partition {partition!r} has no images")
        self._image_size = image_size
        self._augment = augment
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self._records[index]
        try:
            image = np.asarray(Image.open(record.image_path).convert("RGB"), dtype=np.uint8)
        except OSError as exc:
            raise DatasetLoadError(f"cannot read image: {record.image_path}") from exc
        labels = _load_labels(record.label_path)

        if self._augment:
            if self._rng.random() < 0.5 and image.shape[1] > 1:
                image = np.ascontiguousarray(image[:, ::-1, :])
                if labels.size:
                    labels[:, 1] = 1.0 - labels[:, 1]
            image = _hsv_jitter(image, self._rng)

        height, width = image.shape[0], image.shape[1]
        params = _LetterboxParams.compute(width, height, self._image_size)
        canvas = _letterbox_image(image, self._image_size, params)
        tensor = torch.from_numpy(np.ascontiguousarray(canvas.transpose(2, 0, 1))).float() / 255.0

        if labels.size:
            canvas_boxes = labels.copy()
            canvas_boxes[:, 1] = labels[:, 1] * width * params.scale + params.pad_left
            canvas_boxes[:, 2] = labels[:, 2] * height * params.scale + params.pad_top
            canvas_boxes[:, 3] = labels[:, 3] * width * params.scale
            canvas_boxes[:, 4] = labels[:, 4] * height * params.scale
            target = torch.from_numpy(canvas_boxes)
        else:
            target = torch.zeros((0, 5), dtype=torch.float32)
        return tensor, target


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack images; concatenate per-image targets with a prepended batch index column."""
    images = torch.stack([item[0] for item in batch], dim=0)
    indexed: list[torch.Tensor] = []
    for position, (_, boxes) in enumerate(batch):
        if boxes.shape[0]:
            column = torch.full((boxes.shape[0], 1), float(position), dtype=torch.float32)
            indexed.append(torch.cat([column, boxes], dim=1))
    targets = torch.cat(indexed, dim=0) if indexed else torch.zeros((0, 6), dtype=torch.float32)
    return images, targets
