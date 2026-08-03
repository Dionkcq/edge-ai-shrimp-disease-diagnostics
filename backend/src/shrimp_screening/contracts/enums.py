"""Every closed vocabulary the API emits.

Each member's name equals its value. That redundancy is deliberate: it means the
wire format cannot drift from the Python identifier during a rename, and a test
asserts it.
"""

from __future__ import annotations

from enum import StrEnum, unique


@unique
class Decision(StrEnum):
    """The complete set of screening outcomes. There is no sixth member, ever.

    ``UNABLE_TO_ASSESS`` absorbs quality failure, model unavailability, low
    confidence and inference failure. The *reason* lives in
    :class:`AbstentionReason` and :class:`QualityReason`, never in the decision,
    so that no caller can pattern-match a decision string into a health claim.
    """

    GILL_DARKENING_MARKER_DETECTED = "GILL_DARKENING_MARKER_DETECTED"
    WHITE_SPOT_MARKER_DETECTED = "WHITE_SPOT_MARKER_DETECTED"
    MULTIPLE_TARGET_MARKERS_DETECTED = "MULTIPLE_TARGET_MARKERS_DETECTED"
    NO_TARGET_MARKER_DETECTED = "NO_TARGET_MARKER_DETECTED"
    UNABLE_TO_ASSESS = "UNABLE_TO_ASSESS"


@unique
class AbstentionReason(StrEnum):
    """Why the system declined, when and only when the decision is UNABLE_TO_ASSESS."""

    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    IMAGE_QUALITY_REJECTED = "IMAGE_QUALITY_REJECTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    INFERENCE_FAILED = "INFERENCE_FAILED"


@unique
class QualityStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@unique
class QualityReason(StrEnum):
    """Retake instructions. Each maps to one corrective action a person can take."""

    IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL"
    IMAGE_TOO_BLURRY = "IMAGE_TOO_BLURRY"
    IMAGE_TOO_DARK = "IMAGE_TOO_DARK"
    IMAGE_TOO_BRIGHT = "IMAGE_TOO_BRIGHT"
    IMAGE_LOW_CONTRAST = "IMAGE_LOW_CONTRAST"


@unique
class ConfidenceBand(StrEnum):
    """Discrete bands only.

    A percentage would imply a calibrated probability, which no trained model in
    this repository has ever produced.
    """

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    NONE = "NONE"


@unique
class MarkerRole(StrEnum):
    """The screening-relevant meaning of a detected class.

    Roles are assigned by *class name* through the decision policy file, never by
    class index. A class the policy does not name has ``role: null`` and cannot
    influence the decision.
    """

    WHITE_SPOT = "WHITE_SPOT"
    GILL_DARKENING = "GILL_DARKENING"


@unique
class ProviderKind(StrEnum):
    ONNX = "onnx"
    FIXTURE = "fixture"
    UNAVAILABLE = "unavailable"


@unique
class OutputLayout(StrEnum):
    """ONNX output contract. The backend depends on this, not on any trainer."""

    ULTRALYTICS_V8_DETECT_V1 = "ultralytics_v8_detect_v1"
    #: A from-scratch, anchor-based, 3-scale detect head (objectness channel
    #: present). See ``detection/decode.py::decode_custom_yolo_anchor_v1``.
    CUSTOM_YOLO_ANCHOR_V1 = "custom_yolo_anchor_v1"


@unique
class DatasetMappingStatus(StrEnum):
    """Whether the class-index-to-marker mapping of the training data is confirmed.

    ``PROVISIONAL_UNCONFIRMED`` ships in every response for as long as the dataset
    authors have not confirmed the combined-folder class order. It is a
    product-visible fact, not a footnote.
    """

    PROVISIONAL_UNCONFIRMED = "PROVISIONAL_UNCONFIRMED"
    AUTHOR_CONFIRMED = "AUTHOR_CONFIRMED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@unique
class NoticeCode(StrEnum):
    """Response-level facts a user interface must surface, not hide."""

    MODEL_NOT_INSTALLED = "MODEL_NOT_INSTALLED"
    DEMONSTRATION_DATA_NOT_A_REAL_RESULT = "DEMONSTRATION_DATA_NOT_A_REAL_RESULT"
    DATASET_CLASS_MAPPING_UNCONFIRMED = "DATASET_CLASS_MAPPING_UNCONFIRMED"
    THRESHOLDS_UNCALIBRATED = "THRESHOLDS_UNCALIBRATED"


@unique
class ProblemCode(StrEnum):
    """Stable error codes for ``application/problem+json`` responses."""

    MALFORMED_REQUEST = "MALFORMED_REQUEST"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    UNDECODABLE_IMAGE = "UNDECODABLE_IMAGE"
    SERVICE_BUSY = "SERVICE_BUSY"
    NOT_FOUND = "NOT_FOUND"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    ADVICE_UNAVAILABLE = "ADVICE_UNAVAILABLE"
