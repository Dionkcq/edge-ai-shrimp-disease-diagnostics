"""Detections plus a quality verdict, in; exactly one of five decisions, out.

This is the safety boundary of the product, so it is a small pure function over
explicit inputs with no I/O and no hidden state.

Two invariants are enforced here and nowhere else:

* **The decision never carries the reason.** Every way of declining collapses to
  ``UNABLE_TO_ASSESS`` with an :class:`AbstentionReason` beside it, so no client
  can pattern-match a decision string into a health claim.
* **Roles come from the policy, keyed on the model's own class name.** A class the
  policy does not name gets ``role=None`` and cannot move the decision. That is
  what makes a class-order flip in an exported model surface as an unrecognised
  class rather than as a confident mislabel.

Precedence is quality, then availability, then inference failure, then confidence.
An unusable photograph is a retake instruction whether or not a model exists, and
that is the more useful thing to tell someone standing at a pond.
"""

from __future__ import annotations

from dataclasses import dataclass

from shrimp_screening.contracts.enums import (
    AbstentionReason,
    ConfidenceBand,
    Decision,
    MarkerRole,
    QualityStatus,
)
from shrimp_screening.detection.protocol import Detection
from shrimp_screening.policy.loader import DecisionPolicy


@dataclass(frozen=True, slots=True)
class RetainedMarker:
    """A detection that cleared the score threshold, with its policy-assigned role."""

    detection: Detection
    role: MarkerRole | None


@dataclass(frozen=True, slots=True)
class DecisionOutcome:
    """Everything the API needs in order to render one screening."""

    decision: Decision
    abstention_reason: AbstentionReason | None
    confidence_band: ConfidenceBand
    markers: tuple[RetainedMarker, ...]


def _band(policy: DecisionPolicy, best_score: float) -> ConfidenceBand:
    if best_score >= policy.high_score:
        return ConfidenceBand.HIGH
    if best_score >= policy.moderate_score:
        return ConfidenceBand.MODERATE
    return ConfidenceBand.LOW


def _abstention_reason(
    quality: QualityStatus,
    *,
    model_available: bool,
    inference_failed: bool,
) -> AbstentionReason | None:
    """The reason to decline before any detection is looked at, or ``None``.

    The order is the precedence documented at the top of the module: quality, then
    availability, then inference failure.
    """
    if quality is QualityStatus.FAIL:
        return AbstentionReason.IMAGE_QUALITY_REJECTED
    if not model_available:
        return AbstentionReason.MODEL_UNAVAILABLE
    if inference_failed:
        return AbstentionReason.INFERENCE_FAILED
    return None


def decide(
    detections: list[Detection],
    quality: QualityStatus,
    *,
    model_available: bool,
    policy: DecisionPolicy,
    inference_failed: bool = False,
) -> DecisionOutcome:
    """Reduce detections and a quality verdict to one of the five decisions."""
    reason = _abstention_reason(
        quality,
        model_available=model_available,
        inference_failed=inference_failed,
    )
    if reason is not None:
        return DecisionOutcome(Decision.UNABLE_TO_ASSESS, reason, ConfidenceBand.NONE, ())

    above_threshold = [
        RetainedMarker(detection, policy.role_for(detection.class_name))
        for detection in detections
        if detection.score >= policy.minimum_detection_score
    ]
    role_bearing = [marker for marker in above_threshold if marker.role is not None]

    if not role_bearing:
        # Something was seen, but nothing the policy recognises cleared the bar.
        # Reporting "no marker" here would overstate what the model actually said.
        if detections:
            return DecisionOutcome(
                Decision.UNABLE_TO_ASSESS,
                AbstentionReason.LOW_CONFIDENCE,
                ConfidenceBand.LOW,
                tuple(above_threshold),
            )
        return DecisionOutcome(Decision.NO_TARGET_MARKER_DETECTED, None, ConfidenceBand.NONE, ())

    roles = {marker.role for marker in role_bearing}
    band = _band(policy, max(marker.detection.score for marker in role_bearing))
    retained = tuple(above_threshold)

    if len(roles) > 1:
        return DecisionOutcome(Decision.MULTIPLE_TARGET_MARKERS_DETECTED, None, band, retained)
    if MarkerRole.WHITE_SPOT in roles:
        return DecisionOutcome(Decision.WHITE_SPOT_MARKER_DETECTED, None, band, retained)
    return DecisionOutcome(Decision.GILL_DARKENING_MARKER_DETECTED, None, band, retained)
