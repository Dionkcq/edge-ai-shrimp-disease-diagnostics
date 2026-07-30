"""The five approved decision states are a safety boundary, not an implementation detail.

A sixth state -- ``LIKELY_INFECTED``, ``HEALTHY``, ``PROBABLE_WSSV`` -- is the most
plausible way this project would drift into making a claim it cannot support, so the
set is asserted exactly rather than merely sampled.
"""

from __future__ import annotations

from shrimp_screening.contracts.enums import (
    AbstentionReason,
    ConfidenceBand,
    DatasetMappingStatus,
    Decision,
    MarkerRole,
    NoticeCode,
    ProblemCode,
    ProviderKind,
    QualityReason,
    QualityStatus,
)


def test_decision_set_is_exactly_the_five_approved_states() -> None:
    assert {member.value for member in Decision} == {
        "GILL_DARKENING_MARKER_DETECTED",
        "WHITE_SPOT_MARKER_DETECTED",
        "MULTIPLE_TARGET_MARKERS_DETECTED",
        "NO_TARGET_MARKER_DETECTED",
        "UNABLE_TO_ASSESS",
    }


def test_no_decision_asserts_health_or_confirms_a_pathogen() -> None:
    banned = ("HEALTHY", "DISEASE_FREE", "CONFIRMED", "WSSV_POSITIVE", "INFECTED", "DIAGNOS")
    for member in Decision:
        for token in banned:
            assert token not in member.value, f"{member.value} makes an unsupported claim"


def test_abstention_reasons_cover_every_way_the_system_can_decline() -> None:
    assert {member.value for member in AbstentionReason} == {
        "MODEL_UNAVAILABLE",
        "IMAGE_QUALITY_REJECTED",
        "LOW_CONFIDENCE",
        "INFERENCE_FAILED",
    }


def test_quality_reasons_are_actionable_retake_instructions() -> None:
    assert {member.value for member in QualityReason} == {
        "IMAGE_TOO_SMALL",
        "IMAGE_TOO_BLURRY",
        "IMAGE_TOO_DARK",
        "IMAGE_TOO_BRIGHT",
        "IMAGE_LOW_CONTRAST",
    }


def test_problem_codes_are_stable_and_include_the_four_intake_failures() -> None:
    codes = {member.value for member in ProblemCode}
    assert {
        "PAYLOAD_TOO_LARGE",
        "UNSUPPORTED_MEDIA_TYPE",
        "UNDECODABLE_IMAGE",
        "SERVICE_BUSY",
    } <= codes


def test_provider_kinds_include_no_silent_default() -> None:
    assert {member.value for member in ProviderKind} == {"onnx", "fixture", "unavailable"}


def test_fixture_notice_exists_so_demonstration_output_is_never_mistaken_for_a_result() -> None:
    assert NoticeCode.DEMONSTRATION_DATA_NOT_A_REAL_RESULT in NoticeCode
    assert NoticeCode.MODEL_NOT_INSTALLED in NoticeCode


def test_enum_values_are_their_own_names_so_the_wire_format_cannot_drift() -> None:
    for enum_cls in (
        Decision,
        AbstentionReason,
        QualityReason,
        QualityStatus,
        ConfidenceBand,
        NoticeCode,
        ProblemCode,
        MarkerRole,
        DatasetMappingStatus,
    ):
        for member in enum_cls:
            assert member.name == member.value, f"{enum_cls.__name__}.{member.name} drifted"
