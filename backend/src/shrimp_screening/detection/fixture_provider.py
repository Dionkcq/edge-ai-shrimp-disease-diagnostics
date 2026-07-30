"""A detector that replays canned raw tensors through the real decode path.

Why not a mock
--------------
A mock returning ready-made :class:`Detection` objects would exercise none of the
code that can actually be wrong: the letterbox transform, its inverse, the
channels-then-anchors layout assertion, the class-name lookup and NMS. This
provider synthesises a ``(1, 4 + nc, anchors)`` tensor and pushes it through
:func:`decode_ultralytics_v8`, so a demonstration run is a genuine test of every
component except the weights.

It is never a silent default: ``metadata.demonstration`` is ``True``, the API turns
that into a permanent ``DEMONSTRATION_DATA_NOT_A_REAL_RESULT`` notice, and
``settings.py`` refuses to start a demo or production environment on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shrimp_screening.contracts.enums import DatasetMappingStatus, OutputLayout, ProviderKind
from shrimp_screening.detection.decode import decode_ultralytics_v8, expected_anchor_count
from shrimp_screening.detection.letterbox import LetterboxTransform, letterbox_image
from shrimp_screening.detection.protocol import Detection, DetectorMetadata

FIXTURE_INPUT_SIZE = 640

#: Deliberately mirrors the provisional dataset mapping rather than inventing one,
#: so the fixture cannot teach anyone a class order the data does not support.
FIXTURE_CLASS_NAMES: dict[int, str] = {0: "dark_gill", 1: "white_spot"}


@dataclass(frozen=True, slots=True)
class _PlantedBox:
    """A box in normalized original-image coordinates, with a class and a score."""

    class_index: int
    score: float
    box: tuple[float, float, float, float]


#: One canned raw tensor per reachable decision state. Scores straddle the policy
#: thresholds so the low-confidence abstention branch is reachable too.
SCENARIOS: dict[str, tuple[_PlantedBox, ...]] = {
    "none": (),
    "gill": (_PlantedBox(0, 0.84, (0.18, 0.24, 0.38, 0.46)),),
    "white_spot": (_PlantedBox(1, 0.79, (0.54, 0.38, 0.61, 0.46)),),
    "multiple": (
        _PlantedBox(0, 0.84, (0.18, 0.24, 0.38, 0.46)),
        _PlantedBox(1, 0.79, (0.54, 0.38, 0.61, 0.46)),
    ),
    "low_confidence": (_PlantedBox(1, 0.20, (0.54, 0.38, 0.61, 0.46)),),
    #: Two heavily overlapping boxes of one class, so a demo run proves NMS ran.
    "overlapping": (
        _PlantedBox(1, 0.88, (0.50, 0.36, 0.60, 0.46)),
        _PlantedBox(1, 0.71, (0.505, 0.365, 0.605, 0.465)),
    ),
}

DEFAULT_SCENARIO = "multiple"

_METADATA = DetectorMetadata(
    available=True,
    provider=ProviderKind.FIXTURE,
    model_id="fixture-contract-v1",
    version="0.0.0-synthetic",
    output_layout=OutputLayout.ULTRALYTICS_V8_DETECT_V1,
    class_names=dict(FIXTURE_CLASS_NAMES),
    mapping_status=DatasetMappingStatus.PROVISIONAL_UNCONFIRMED,
    demonstration=True,
)


class UnknownScenarioError(ValueError):
    """A scenario name that the fixture corpus does not define."""


def build_raw_output(
    planted: tuple[_PlantedBox, ...],
    transform: LetterboxTransform,
    *,
    class_count: int = len(FIXTURE_CLASS_NAMES),
    input_size: int = FIXTURE_INPUT_SIZE,
) -> np.ndarray:
    """Synthesise a ``(1, 4 + nc, anchors)`` detect-head tensor.

    Background anchors carry a small non-zero score, matching a real sigmoid head,
    so the score threshold is genuinely exercised rather than trivially satisfied.
    """
    anchors = expected_anchor_count(input_size)
    raw = np.zeros((1, 4 + class_count, anchors), dtype=np.float32)
    raw[0, 4:, :] = 0.01
    for slot, item in enumerate(planted):
        x1, y1, x2, y2 = transform.forward_xyxy(
            (
                item.box[0] * transform.original_width,
                item.box[1] * transform.original_height,
                item.box[2] * transform.original_width,
                item.box[3] * transform.original_height,
            )
        )
        raw[0, 0, slot] = (x1 + x2) / 2.0
        raw[0, 1, slot] = (y1 + y2) / 2.0
        raw[0, 2, slot] = x2 - x1
        raw[0, 3, slot] = y2 - y1
        raw[0, 4 + item.class_index, slot] = item.score
    return raw


class FixtureProvider:
    """Replays one canned scenario through the production decode path."""

    def __init__(
        self,
        *,
        score_threshold: float,
        iou_threshold: float,
        max_detections: int = 300,
        scenario: str = DEFAULT_SCENARIO,
    ) -> None:
        if scenario not in SCENARIOS:
            raise UnknownScenarioError(
                f"unknown fixture scenario {scenario!r}; choose one of {sorted(SCENARIOS)}"
            )
        self._score_threshold = score_threshold
        self._iou_threshold = iou_threshold
        self._max_detections = max_detections
        self._scenario = scenario

    @property
    def metadata(self) -> DetectorMetadata:
        return _METADATA

    @property
    def scenario(self) -> str:
        return self._scenario

    def with_scenario(self, scenario: str) -> FixtureProvider:
        """Return a sibling provider replaying a different canned tensor."""
        return FixtureProvider(
            score_threshold=self._score_threshold,
            iou_threshold=self._iou_threshold,
            max_detections=self._max_detections,
            scenario=scenario,
        )

    def infer(self, image: np.ndarray) -> list[Detection]:
        # The image is letterboxed for real: the transform used to plant the boxes
        # is the same one used to recover them, so a regression in either direction
        # shows up as a moved box rather than as a passing test.
        _, transform = letterbox_image(image, FIXTURE_INPUT_SIZE)
        raw = build_raw_output(SCENARIOS[self._scenario], transform)
        return decode_ultralytics_v8(
            raw,
            FIXTURE_CLASS_NAMES,
            transform,
            score_threshold=self._score_threshold,
            iou_threshold=self._iou_threshold,
            max_detections=self._max_detections,
        )
