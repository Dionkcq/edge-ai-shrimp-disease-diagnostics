# SPDX-License-Identifier: AGPL-3.0-or-later
"""Dataset-derived anchor box priors, computed once per training run.

Standard k-means-over-box-statistics ("autoanchor"), in plain numpy so it stays a
dependency-free, deterministic, unit-testable step. Anchors are *recorded*, never
re-derived at inference time: the runtime decoder
(``backend/src/shrimp_screening/detection/decode.py``) receives the exact 9
``(width, height)`` pairs computed here through the model registry entry, so a
screening result stays reproducible from the same registered artifact.

Boxes are approximated into 640x640 canvas space via ``normalized_w * 640`` /
``normalized_h * 640``, ignoring each image's own letterbox padding. This is a
deliberate simplification -- the same one most from-scratch YOLO implementations
use -- acceptable because anchors are a statistical prior, not a safety-critical
measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from shrimp_training.dataset import PreparedDataset
from shrimp_training.model import STRIDES

#: Anchors per detection scale. 3 scales x 3 anchors/scale = 9 total, matching
#: ``model.YOLO(num_anchors=3)``.
ANCHORS_PER_SCALE = 3
CANVAS_SIZE = 640
_MINIMUM_BOX_PIXELS = 1.0


class AnchorError(RuntimeError):
    """The prepared dataset does not contain enough boxes to derive anchors."""


@dataclass(frozen=True, slots=True)
class AnchorSet:
    """9 (width, height) pixel pairs in 640-canvas space, ordered scale-major.

    ``boxes[0:3]`` pair with ``strides[0]`` (8, finest/smallest anchors),
    ``boxes[3:6]`` with ``strides[1]`` (16), ``boxes[6:9]`` with ``strides[2]`` (32).
    """

    boxes: tuple[tuple[float, float], ...]
    strides: tuple[int, ...]
    seed: int
    source_box_count: int

    def for_stride(self, stride: int) -> tuple[tuple[float, float], ...]:
        index = self.strides.index(stride)
        return self.boxes[index * ANCHORS_PER_SCALE : (index + 1) * ANCHORS_PER_SCALE]


def _read_boxes(manifest_path: Path) -> list[tuple[float, float]]:
    """Read every training-partition box's (width, height) in 640-canvas pixels."""
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AnchorError(f"cannot read prepared manifest: {manifest_path}") from exc
    root = manifest_path.parent
    boxes: list[tuple[float, float]] = []
    for record in document.get("images", []):
        if record.get("partition") != "train":
            continue
        label_path = root / str(record["label_output"])
        try:
            lines = label_path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise AnchorError(f"cannot read label file: {label_path}") from exc
        for line in lines:
            fields = line.split()
            if len(fields) != 5:
                continue
            width, height = float(fields[3]), float(fields[4])
            boxes.append((width * CANVAS_SIZE, height * CANVAS_SIZE))
    return boxes


def _iou_matrix(points: NDArray[Any], centroids: NDArray[Any]) -> NDArray[Any]:
    """Shape-only IoU (both boxes centered at the origin) between every pair.

    Scale-aware in a way plain Euclidean distance on ``(w, h)`` is not -- a large
    box and a small box of the same aspect ratio are still far apart.
    """
    point_w, point_h = points[:, 0], points[:, 1]
    centroid_w, centroid_h = centroids[:, 0], centroids[:, 1]
    min_w = np.minimum(point_w[:, None], centroid_w[None, :])
    min_h = np.minimum(point_h[:, None], centroid_h[None, :])
    intersection = min_w * min_h
    point_area = (point_w * point_h)[:, None]
    centroid_area = (centroid_w * centroid_h)[None, :]
    union = point_area + centroid_area - intersection
    result: NDArray[Any] = intersection / np.maximum(union, 1e-9)
    return result


def _kmeans(points: NDArray[Any], k: int, *, seed: int, iterations: int = 200) -> NDArray[Any]:
    """Deterministic k-means in IoU-as-distance space (Ultralytics-style autoanchor)."""
    rng = np.random.default_rng(seed)
    centroids = points[rng.integers(0, len(points), size=1)].copy()
    while len(centroids) < k:
        distances = 1.0 - _iou_matrix(points, centroids).max(axis=1)
        centroids = np.vstack([centroids, points[[int(np.argmax(distances))]]])

    for _ in range(iterations):
        assignments = _iou_matrix(points, centroids).argmax(axis=1)
        new_centroids = centroids.copy()
        for cluster in range(k):
            members = points[assignments == cluster]
            if len(members):
                new_centroids[cluster] = members.mean(axis=0)
        if np.allclose(new_centroids, centroids):
            centroids = new_centroids
            break
        centroids = new_centroids
    return centroids


def compute_anchors(prepared: PreparedDataset, *, seed: int = 20260730) -> AnchorSet:
    """Derive 9 anchor boxes from the training partition's own box statistics."""
    boxes = _read_boxes(prepared.manifest)
    required = len(STRIDES) * ANCHORS_PER_SCALE
    if len(boxes) < required:
        raise AnchorError(
            f"the training partition has only {len(boxes)} boxes; at least "
            f"{required} are required to derive {required} anchors"
        )
    points = np.asarray(boxes, dtype=np.float64)
    centroids = _kmeans(points, required, seed=seed)
    order = np.argsort(centroids[:, 0] * centroids[:, 1])
    sorted_centroids = centroids[order]
    boxes_out = tuple(
        (float(max(w, _MINIMUM_BOX_PIXELS)), float(max(h, _MINIMUM_BOX_PIXELS)))
        for w, h in sorted_centroids
    )
    return AnchorSet(boxes=boxes_out, strides=STRIDES, seed=seed, source_box_count=len(boxes))


def anchor_set_to_document(anchors: AnchorSet) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "strides": list(anchors.strides),
        "anchors_per_scale": ANCHORS_PER_SCALE,
        "boxes": [list(box) for box in anchors.boxes],
        "seed": anchors.seed,
        "source_box_count": anchors.source_box_count,
    }
