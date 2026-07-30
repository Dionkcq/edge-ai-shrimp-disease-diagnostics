"""The screening response contract. This module is the single source of truth.

`contracts/screening_result.schema.json` is generated from these models and committed;
a drift test regenerates and byte-compares it. Frontend types will be generated from
that schema in a later slice, so any change here is a deliberate contract change.

Versioning rules live in `contracts/CONTRACT.md`.
"""

from __future__ import annotations

from typing import Annotated, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from shrimp_screening.contracts.enums import (
    AbstentionReason,
    ConfidenceBand,
    DatasetMappingStatus,
    Decision,
    MarkerRole,
    NoticeCode,
    OutputLayout,
    ProviderKind,
    QualityReason,
    QualityStatus,
)

#: Bumped on any breaking change to the response shape. See contracts/CONTRACT.md.
#: Typed as the literal it is so that the `schema_version` field default and the
#: exported JSON Schema's `const` cannot drift apart without a type error.
SCHEMA_VERSION: Final[Literal["1.0.0"]] = "1.0.0"

#: `model_*` is a protected namespace in Pydantic v2. `model.model_id` and the
#: top-level `model` object are part of the published contract, so the protection
#: is switched off rather than the contract being renamed around it.
_STRICT = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

NormalizedCoordinate = Annotated[float, Field(ge=0.0, le=1.0)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]
Milliseconds = Annotated[float, Field(ge=0.0)]


class BoundingBox(BaseModel):
    """Normalized xyxy in the EXIF-corrected *original* image frame.

    Normalized, so a client that scales or letterboxes the preview cannot drift from
    the overlay. Original frame and post-transpose, so a client never has to reason
    about orientation.
    """

    model_config = _STRICT

    x1: NormalizedCoordinate
    y1: NormalizedCoordinate
    x2: NormalizedCoordinate
    y2: NormalizedCoordinate

    @model_validator(mode="after")
    def _corners_are_ordered(self) -> Self:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("bounding box corners must satisfy x1 <= x2 and y1 <= y2")
        return self


class MarkerObservation(BaseModel):
    """One retained detection.

    ``class_name`` comes from the model's own metadata; ``role`` comes from the
    decision policy keyed on that name. No index-to-label table exists anywhere in
    the backend, so a class-order flip in an exported model changes the labels in the
    response instead of silently mislabelling them.
    """

    model_config = _STRICT

    class_index: int = Field(ge=0)
    class_name: str = Field(min_length=1, max_length=64)
    role: MarkerRole | None = Field(
        default=None,
        description="Screening role assigned by the decision policy, or null if the "
        "policy does not recognise this class name.",
    )
    score: Score
    box: BoundingBox


class QualityMetrics(BaseModel):
    """Raw measurements, echoed so a retake instruction is auditable."""

    model_config = _STRICT

    blur_score: float = Field(ge=0.0, description="Variance of the Laplacian response.")
    mean_luminance: float = Field(ge=0.0, le=255.0)
    rms_contrast: float = Field(ge=0.0)
    min_side_px: int = Field(ge=1)


class QualityReport(BaseModel):
    model_config = _STRICT

    status: QualityStatus
    reasons: list[QualityReason] = Field(default_factory=list)
    metrics: QualityMetrics
    policy_id: str
    policy_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _status_agrees_with_reasons(self) -> Self:
        if self.status is QualityStatus.FAIL and not self.reasons:
            raise ValueError("a quality failure must carry at least one retake reason")
        if self.status is QualityStatus.PASS and self.reasons:
            raise ValueError("a passing image must not carry retake reasons")
        return self


class ImageInfo(BaseModel):
    """Post-EXIF-transpose facts about the decoded image.

    Deliberately excludes every field that could carry personal data: no filename, no
    EXIF block, no GPS, no capture timestamp, no camera make or model.
    """

    model_config = _STRICT

    width: int = Field(ge=1)
    height: int = Field(ge=1)
    source_mode: str = Field(
        min_length=1,
        max_length=16,
        description="Pillow mode of the file as decoded, before RGB normalization. "
        "Echoed so an inverted CMYK conversion is debuggable.",
    )
    exif_transposed: bool


class ModelInfo(BaseModel):
    model_config = _STRICT

    available: bool
    provider: ProviderKind
    model_id: str | None = None
    version: str | None = None
    output_layout: OutputLayout | None = None
    class_names: dict[str, str] = Field(
        default_factory=dict,
        description="Class index (as a string key) to class name, read from the model "
        "artifact's own metadata. Drives all labelling.",
    )
    dataset_mapping_status: DatasetMappingStatus
    is_demonstration_data: bool = Field(
        description="True when the numbers in this response were synthesised for "
        "demonstration and are not a real model result."
    )

    @model_validator(mode="after")
    def _demonstration_and_availability_agree_with_provider(self) -> Self:
        if self.provider is ProviderKind.UNAVAILABLE and self.available:
            raise ValueError("the unavailable provider cannot report an available model")
        if self.provider is ProviderKind.FIXTURE and not self.is_demonstration_data:
            raise ValueError("fixture output must always be flagged as demonstration data")
        if self.provider is ProviderKind.ONNX and self.is_demonstration_data:
            raise ValueError("a real ONNX result must not be flagged as demonstration data")
        return self


class Timings(BaseModel):
    model_config = _STRICT

    intake_ms: Milliseconds
    quality_ms: Milliseconds
    inference_ms: Milliseconds
    total_ms: Milliseconds


class ScreeningResult(BaseModel):
    """The response body of ``POST /api/v1/screenings``."""

    model_config = _STRICT

    schema_version: Literal["1.0.0"] = SCHEMA_VERSION
    request_id: str = Field(
        pattern=r"^[0-9A-HJKMNP-TV-Z]{26}$",
        description="ULID. The only identifier that appears in server logs.",
    )
    decision: Decision
    abstention_reason: AbstentionReason | None = None
    quality: QualityReport
    markers: list[MarkerObservation] = Field(default_factory=list)
    confidence_band: ConfidenceBand
    image: ImageInfo
    model: ModelInfo
    notices: list[NoticeCode] = Field(default_factory=list)
    guidance_refs: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(
        default_factory=list,
        description="Identifiers defined in docs/LIMITATIONS.md.",
    )
    timings_ms: Timings

    @model_validator(mode="after")
    def _abstention_reason_is_present_exactly_when_abstaining(self) -> Self:
        abstaining = self.decision is Decision.UNABLE_TO_ASSESS
        if abstaining and self.abstention_reason is None:
            raise ValueError("UNABLE_TO_ASSESS must state why")
        if not abstaining and self.abstention_reason is not None:
            raise ValueError("abstention_reason is only valid for UNABLE_TO_ASSESS")
        return self

    @model_validator(mode="after")
    def _a_positive_decision_requires_a_retained_marker(self) -> Self:
        marker_decisions = {
            Decision.WHITE_SPOT_MARKER_DETECTED,
            Decision.GILL_DARKENING_MARKER_DETECTED,
            Decision.MULTIPLE_TARGET_MARKERS_DETECTED,
        }
        if self.decision in marker_decisions and not any(m.role for m in self.markers):
            raise ValueError(f"{self.decision} requires at least one role-bearing marker")
        return self

    @model_validator(mode="after")
    def _unavailable_model_cannot_report_markers(self) -> Self:
        if not self.model.available and self.markers:
            raise ValueError("an unavailable model cannot have produced markers")
        return self
