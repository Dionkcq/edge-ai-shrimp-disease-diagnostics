"""The quality gate is deterministic and runs before any model exists.

The property that matters most is monotonicity: a photograph that is made steadily
worse must never cross back from FAIL to PASS. A threshold gate assembled from
independent measurements can violate that surprisingly easily.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageFilter

from shrimp_screening.contracts.enums import QualityReason, QualityStatus
from shrimp_screening.imaging.quality import assess_quality, variance_of_laplacian
from shrimp_screening.policy.loader import PolicyError, QualityPolicy, load_quality_policy

POLICY = QualityPolicy(
    policy_id="test_quality",
    policy_hash="sha256:" + "0" * 64,
    minimum_side_px=640,
    minimum_blur_score=18.0,
    minimum_mean_luminance=25.0,
    maximum_mean_luminance=235.0,
    minimum_rms_contrast=12.0,
    status="UNCALIBRATED",
)


def textured(width: int = 800, height: int = 800) -> np.ndarray:
    yy, xx = np.indices((height, width))
    plane = np.clip((xx * 7 + yy * 11) % 180 + 35, 0, 255).astype(np.uint8)
    return np.repeat(plane[:, :, None], 3, axis=2)


def test_a_detailed_well_lit_image_passes() -> None:
    report = assess_quality(textured(), POLICY)
    assert report.status is QualityStatus.PASS
    assert report.reasons == []


def test_a_black_frame_fails_for_darkness_and_for_flatness() -> None:
    report = assess_quality(np.zeros((800, 800, 3), dtype=np.uint8), POLICY)
    assert report.status is QualityStatus.FAIL
    assert QualityReason.IMAGE_TOO_DARK in report.reasons
    assert QualityReason.IMAGE_LOW_CONTRAST in report.reasons


def test_a_blown_out_frame_is_reported_as_too_bright() -> None:
    report = assess_quality(np.full((800, 800, 3), 250, dtype=np.uint8), POLICY)
    assert QualityReason.IMAGE_TOO_BRIGHT in report.reasons


def test_an_undersized_photograph_is_rejected_before_it_can_be_upscaled() -> None:
    report = assess_quality(textured(400, 400), POLICY)
    assert QualityReason.IMAGE_TOO_SMALL in report.reasons


def test_a_flat_frame_fails_for_blur_even_when_correctly_exposed() -> None:
    report = assess_quality(np.full((800, 800, 3), 128, dtype=np.uint8), POLICY)
    assert QualityReason.IMAGE_TOO_BLURRY in report.reasons
    assert report.metrics.mean_luminance == pytest.approx(128.0, abs=0.5)


def test_increasing_blur_never_flips_a_failure_back_into_a_pass() -> None:
    source = Image.fromarray(textured())
    previous_blur = float("inf")
    seen_failure = False
    for radius in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
        blurred = np.asarray(
            source.filter(ImageFilter.GaussianBlur(radius)) if radius else source, dtype=np.uint8
        )
        report = assess_quality(blurred, POLICY)
        assert report.metrics.blur_score <= previous_blur + 1e-6, "blur score must be monotone"
        previous_blur = report.metrics.blur_score
        if report.status is QualityStatus.FAIL:
            seen_failure = True
        else:
            assert not seen_failure, f"radius {radius} recovered a PASS after a FAIL"
    assert seen_failure, "the ladder never reached a failing blur; the fixture is too sharp"


def test_status_and_reasons_cannot_disagree() -> None:
    passing = assess_quality(textured(), POLICY)
    assert (passing.status is QualityStatus.PASS) == (not passing.reasons)
    failing = assess_quality(np.zeros((100, 100, 3), dtype=np.uint8), POLICY)
    assert (failing.status is QualityStatus.FAIL) == bool(failing.reasons)


def test_luminance_is_perceptual_not_a_flat_channel_mean() -> None:
    """A flat mean would over-weight blue, which carries the least detail and the
    most sensor noise in dim pond light."""
    blue = np.zeros((640, 640, 3), dtype=np.uint8)
    blue[:, :, 2] = 255
    green = np.zeros((640, 640, 3), dtype=np.uint8)
    green[:, :, 1] = 255
    assert (
        assess_quality(green, POLICY).metrics.mean_luminance
        > assess_quality(blue, POLICY).metrics.mean_luminance
    )


def test_a_tiny_image_does_not_crash_the_laplacian() -> None:
    assert variance_of_laplacian(np.zeros((2, 2), dtype=np.float32)) == 0.0


def test_a_non_rgb_array_is_rejected_rather_than_silently_measured() -> None:
    with pytest.raises(ValueError, match="RGB"):
        assess_quality(np.zeros((10, 10), dtype=np.uint8), POLICY)


def test_the_shipped_policy_loads_and_is_hashed() -> None:
    policy = load_quality_policy()
    assert policy.policy_id == "quality_policy_v1"
    assert policy.policy_hash.startswith("sha256:")
    assert len(policy.policy_hash) == len("sha256:") + 64
    assert policy.status == "UNCALIBRATED", "no threshold here has ever been measured"


def test_a_policy_with_inverted_luminance_bounds_is_rejected() -> None:
    with pytest.raises(PolicyError, match="minimum_mean_luminance"):
        QualityPolicy(
            policy_id="broken",
            policy_hash="sha256:" + "0" * 64,
            minimum_side_px=640,
            minimum_blur_score=18.0,
            minimum_mean_luminance=200.0,
            maximum_mean_luminance=100.0,
            minimum_rms_contrast=12.0,
            status="UNCALIBRATED",
        )
