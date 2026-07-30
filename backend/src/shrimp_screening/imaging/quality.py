"""The deterministic, model-free retake gate.

Runs before any detector and produces an actionable instruction rather than a
score. "Too dark -- move into shade with more ambient light" is something a person
holding a phone can act on; "quality 0.42" is not.

Three measurements, all numpy, no OpenCV:

* **blur** -- variance of the Laplacian response. High-frequency energy collapses
  when a photograph is out of focus or motion-blurred. The 3x3 four-neighbour
  kernel is applied by shifted addition, and the one-pixel border is discarded
  because it has no complete neighbourhood.
* **mean luminance** -- catches under- and over-exposure, both of which destroy the
  low-contrast gill shading this project needs to see.
* **RMS contrast** -- catches a flat, washed-out or fogged frame that is neither
  dark nor bright but carries no usable signal.

Deliberately *not* included: any judgement about whether the photograph contains a
shrimp. There is no out-of-distribution gate in this cycle, and pretending
otherwise here would be worse than the honest gap recorded in docs/KNOWN_GAPS.md.
"""

from __future__ import annotations

import numpy as np

from shrimp_screening.contracts.enums import QualityReason, QualityStatus
from shrimp_screening.contracts.screening import QualityMetrics, QualityReport
from shrimp_screening.policy.loader import QualityPolicy

#: A Laplacian needs a complete 3x3 neighbourhood, so anything smaller than this
#: has no interior to measure.
_MINIMUM_MEASURABLE_SIDE = 3


def _luminance(image: np.ndarray) -> np.ndarray:
    """Rec. 601 luma. A plain channel mean would over-weight blue, which carries
    the least perceptual detail and the most sensor noise in dim pond light."""
    channels = image.astype(np.float32)
    luma: np.ndarray = (
        0.299 * channels[:, :, 0] + 0.587 * channels[:, :, 1] + 0.114 * channels[:, :, 2]
    )
    return luma


def variance_of_laplacian(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian over the interior of the image."""
    if min(gray.shape) < _MINIMUM_MEASURABLE_SIDE:
        return 0.0
    response = -4.0 * gray
    response[1:, :] += gray[:-1, :]
    response[:-1, :] += gray[1:, :]
    response[:, 1:] += gray[:, :-1]
    response[:, :-1] += gray[:, 1:]
    return float(response[1:-1, 1:-1].var())


def assess_quality(image: np.ndarray, policy: QualityPolicy) -> QualityReport:
    """Measure one decoded RGB image and return a pass/fail with retake reasons."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected an (h, w, 3) RGB array, got shape {image.shape}")

    gray = _luminance(image)
    blur = variance_of_laplacian(gray)
    mean_luminance = float(gray.mean())
    rms_contrast = float(gray.std())
    min_side = int(min(image.shape[0], image.shape[1]))

    reasons: list[QualityReason] = []
    if min_side < policy.minimum_side_px:
        reasons.append(QualityReason.IMAGE_TOO_SMALL)
    if blur < policy.minimum_blur_score:
        reasons.append(QualityReason.IMAGE_TOO_BLURRY)
    if mean_luminance < policy.minimum_mean_luminance:
        reasons.append(QualityReason.IMAGE_TOO_DARK)
    if mean_luminance > policy.maximum_mean_luminance:
        reasons.append(QualityReason.IMAGE_TOO_BRIGHT)
    if rms_contrast < policy.minimum_rms_contrast:
        reasons.append(QualityReason.IMAGE_LOW_CONTRAST)

    return QualityReport(
        status=QualityStatus.FAIL if reasons else QualityStatus.PASS,
        reasons=reasons,
        metrics=QualityMetrics(
            blur_score=max(0.0, blur),
            mean_luminance=min(255.0, max(0.0, mean_luminance)),
            rms_contrast=max(0.0, rms_contrast),
            min_side_px=min_side,
        ),
        policy_id=policy.policy_id,
        policy_hash=policy.policy_hash,
    )
