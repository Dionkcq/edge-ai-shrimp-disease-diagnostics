"""The five decisions and every route to abstention.

Table-driven on purpose: the interesting property is that the mapping from inputs
to decisions is *total and closed*, and a table makes an accidentally unreachable
state visible as a missing row.
"""

from __future__ import annotations

import pytest

from shrimp_screening.contracts.enums import (
    AbstentionReason,
    ConfidenceBand,
    Decision,
    MarkerRole,
    QualityStatus,
)
from shrimp_screening.detection.protocol import Detection
from shrimp_screening.policy.decision import decide
from shrimp_screening.policy.loader import DecisionPolicy, PolicyError, load_decision_policy

POLICY = DecisionPolicy(
    policy_id="test_policy",
    policy_hash="sha256:" + "0" * 64,
    candidate_detection_score=0.15,
    minimum_detection_score=0.35,
    moderate_score=0.55,
    high_score=0.80,
    iou_threshold=0.45,
    max_detections=300,
    class_roles={"dark_gill": MarkerRole.GILL_DARKENING, "white_spot": MarkerRole.WHITE_SPOT},
    status="UNCALIBRATED",
)


def detection(name: str, score: float = 0.9, index: int = 0) -> Detection:
    return Detection(index, name, score, (0.1, 0.1, 0.2, 0.2))


GILL = detection("dark_gill", index=0)
SPOT = detection("white_spot", index=1)


@pytest.mark.parametrize(
    ("detections", "quality", "available", "expected"),
    [
        ([], QualityStatus.PASS, True, Decision.NO_TARGET_MARKER_DETECTED),
        ([GILL], QualityStatus.PASS, True, Decision.GILL_DARKENING_MARKER_DETECTED),
        ([SPOT], QualityStatus.PASS, True, Decision.WHITE_SPOT_MARKER_DETECTED),
        ([GILL, SPOT], QualityStatus.PASS, True, Decision.MULTIPLE_TARGET_MARKERS_DETECTED),
        ([SPOT, GILL], QualityStatus.PASS, True, Decision.MULTIPLE_TARGET_MARKERS_DETECTED),
        ([GILL], QualityStatus.FAIL, True, Decision.UNABLE_TO_ASSESS),
        ([], QualityStatus.PASS, False, Decision.UNABLE_TO_ASSESS),
    ],
)
def test_every_decision_state_is_reachable(
    detections: list[Detection],
    quality: QualityStatus,
    available: bool,
    expected: Decision,
) -> None:
    outcome = decide(detections, quality, model_available=available, policy=POLICY)
    assert outcome.decision is expected


@pytest.mark.parametrize(
    ("kwargs", "quality", "detections", "expected"),
    [
        (
            {"model_available": True},
            QualityStatus.FAIL,
            [],
            AbstentionReason.IMAGE_QUALITY_REJECTED,
        ),
        ({"model_available": False}, QualityStatus.PASS, [], AbstentionReason.MODEL_UNAVAILABLE),
        (
            {"model_available": True, "inference_failed": True},
            QualityStatus.PASS,
            [],
            AbstentionReason.INFERENCE_FAILED,
        ),
        (
            {"model_available": True},
            QualityStatus.PASS,
            [detection("white_spot", 0.2)],
            AbstentionReason.LOW_CONFIDENCE,
        ),
    ],
)
def test_every_abstention_branch_names_its_reason(
    kwargs: dict[str, bool],
    quality: QualityStatus,
    detections: list[Detection],
    expected: AbstentionReason,
) -> None:
    outcome = decide(detections, quality, policy=POLICY, **kwargs)
    assert outcome.decision is Decision.UNABLE_TO_ASSESS
    assert outcome.abstention_reason is expected


def test_quality_failure_outranks_model_unavailability() -> None:
    """A retake instruction is actionable; "no model installed" is not."""
    outcome = decide([], QualityStatus.FAIL, model_available=False, policy=POLICY)
    assert outcome.abstention_reason is AbstentionReason.IMAGE_QUALITY_REJECTED


def test_a_non_abstaining_decision_carries_no_reason() -> None:
    outcome = decide([SPOT], QualityStatus.PASS, model_available=True, policy=POLICY)
    assert outcome.abstention_reason is None


def test_an_unrecognised_class_name_cannot_move_the_decision() -> None:
    """This is the defence against a class-order flip in an exported model.

    A class the policy does not name gets no role, so it is reported to the user
    but cannot be turned into a marker decision.
    """
    outcome = decide(
        [detection("tail_discoloration", 0.99)],
        QualityStatus.PASS,
        model_available=True,
        policy=POLICY,
    )
    assert outcome.decision is Decision.UNABLE_TO_ASSESS
    assert outcome.abstention_reason is AbstentionReason.LOW_CONFIDENCE
    assert [marker.role for marker in outcome.markers] == [None]


def test_nothing_detected_at_all_is_not_the_same_as_nothing_retained() -> None:
    """An empty detector output is "no marker"; a weak output is an abstention."""
    assert (
        decide([], QualityStatus.PASS, model_available=True, policy=POLICY).decision
        is Decision.NO_TARGET_MARKER_DETECTED
    )
    assert (
        decide(
            [detection("white_spot", 0.1)],
            QualityStatus.PASS,
            model_available=True,
            policy=POLICY,
        ).decision
        is Decision.UNABLE_TO_ASSESS
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.95, ConfidenceBand.HIGH),
        (0.80, ConfidenceBand.HIGH),
        (0.79, ConfidenceBand.MODERATE),
        (0.55, ConfidenceBand.MODERATE),
        (0.54, ConfidenceBand.LOW),
        (0.35, ConfidenceBand.LOW),
    ],
)
def test_confidence_bands_follow_the_policy_boundaries(
    score: float, expected: ConfidenceBand
) -> None:
    outcome = decide(
        [detection("white_spot", score)], QualityStatus.PASS, model_available=True, policy=POLICY
    )
    assert outcome.confidence_band is expected


def test_abstention_never_reports_a_confidence_band_above_low() -> None:
    for outcome in (
        decide([], QualityStatus.FAIL, model_available=True, policy=POLICY),
        decide([], QualityStatus.PASS, model_available=False, policy=POLICY),
    ):
        assert outcome.confidence_band is ConfidenceBand.NONE


def test_an_unavailable_model_reports_no_markers_at_all() -> None:
    outcome = decide([SPOT], QualityStatus.PASS, model_available=False, policy=POLICY)
    assert outcome.markers == ()


def test_policy_rejects_thresholds_that_are_out_of_order() -> None:
    with pytest.raises(PolicyError, match="scores must satisfy"):
        DecisionPolicy(
            policy_id="broken",
            policy_hash="sha256:" + "0" * 64,
            candidate_detection_score=0.15,
            minimum_detection_score=0.9,
            moderate_score=0.5,
            high_score=0.8,
            iou_threshold=0.45,
            max_detections=300,
            class_roles={"white_spot": MarkerRole.WHITE_SPOT},
            status="UNCALIBRATED",
        )


def test_policy_rejects_a_candidate_floor_above_the_reporting_bar() -> None:
    """If the decode floor were above the reporting bar, nothing would ever be reported."""
    with pytest.raises(PolicyError, match="scores must satisfy"):
        DecisionPolicy(
            policy_id="broken",
            policy_hash="sha256:" + "0" * 64,
            candidate_detection_score=0.5,
            minimum_detection_score=0.35,
            moderate_score=0.55,
            high_score=0.8,
            iou_threshold=0.45,
            max_detections=300,
            class_roles={"white_spot": MarkerRole.WHITE_SPOT},
            status="UNCALIBRATED",
        )


def test_the_shipped_policy_leaves_room_for_a_low_confidence_abstention() -> None:
    """A regression guard on the shipped file, not on a hand-built fixture.

    If someone raises `candidate_detection_score` to equal `minimum_detection_score`,
    every weak sighting is discarded during decode and the LOW_CONFIDENCE branch
    silently becomes unreachable -- the system would answer NO_TARGET_MARKER_DETECTED
    for an image the model was uncertain about. This test fails if that happens.
    """
    shipped = load_decision_policy()
    assert shipped.candidate_detection_score < shipped.minimum_detection_score, (
        "the shipped decision policy must keep the decode floor strictly below the "
        "reporting bar, or LOW_CONFIDENCE is dead code"
    )
