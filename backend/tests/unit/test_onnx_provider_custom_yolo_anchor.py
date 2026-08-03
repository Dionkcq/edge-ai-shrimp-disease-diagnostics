"""The load-time contract and inference dispatch for ``CUSTOM_YOLO_ANCHOR_V1``.

Uses fake session objects rather than a real ONNX graph (unlike
``test_onnx_provider.py``, which builds real graphs for the Ultralytics path) --
the point here is ``_validate_session``'s branch-selection logic and
``OnnxProvider.infer()``'s decoder dispatch, both of which only need something
that answers ``get_inputs``/``get_outputs``/``get_modelmeta``/``run`` the way
onnxruntime does.
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pytest

from shrimp_screening.contracts.enums import DatasetMappingStatus, OutputLayout
from shrimp_screening.detection.decode import expected_anchor_count
from shrimp_screening.detection.onnx_provider import (
    ModelContractError,
    OnnxProvider,
    _validate_session,
)
from shrimp_screening.detection.registry import RegisteredModel

INPUT_SIZE = 64
CLASS_NAMES = {0: "dark_gill", 1: "white_spot"}
ANCHORS = tuple((2.0 * (index + 1), 2.0 * (index + 1)) for index in range(9))
TOTAL_ANCHOR_POSITIONS = 3 * expected_anchor_count(INPUT_SIZE)


def _entry(**overrides: object) -> RegisteredModel:
    base: dict[str, object] = {
        "model_id": "synthetic-custom",
        "version": "0.0.0-synthetic",
        "filename": "synthetic-custom.onnx",
        "sha256": "a" * 64,
        "input_size": INPUT_SIZE,
        "class_names": dict(CLASS_NAMES),
        "opset": 17,
        "output_layout": OutputLayout.CUSTOM_YOLO_ANCHOR_V1,
        "anchors": ANCHORS,
        "dataset_mapping_status": DatasetMappingStatus.PROVISIONAL_UNCONFIRMED,
        "artifact_license": "AGPL-3.0-or-later",
        "training_toolchain": "custom-pytorch-yolo torch 2.5.1",
    }
    base.update(overrides)
    return RegisteredModel(**base)  # type: ignore[arg-type]


class _Node:
    def __init__(
        self, shape: list[int], node_type: str = "tensor(float)", name: str = "images"
    ) -> None:
        self.shape = shape
        self.type = node_type
        self.name = name


class _Metadata:
    def __init__(self, custom_metadata_map: dict[str, str]) -> None:
        self.custom_metadata_map = custom_metadata_map


def _default_metadata() -> dict[str, str]:
    return {
        "names": repr(CLASS_NAMES),
        "task": "detect",
        "imgsz": f"[{INPUT_SIZE}, {INPUT_SIZE}]",
    }


class _FakeSession:
    output_shape: ClassVar[list[int]]

    def __init__(
        self,
        *,
        output_shape: list[int] | None = None,
        metadata: dict[str, str] | None = None,
        run_result: np.ndarray | None = None,
    ) -> None:
        self.output_shape = output_shape or [1, 7, TOTAL_ANCHOR_POSITIONS]
        self._metadata = _default_metadata() if metadata is None else metadata
        self._run_result = run_result

    def get_inputs(self) -> list[_Node]:
        return [_Node([1, 3, INPUT_SIZE, INPUT_SIZE])]

    def get_outputs(self) -> list[_Node]:
        return [_Node(self.output_shape)]

    def get_modelmeta(self) -> _Metadata:
        return _Metadata(self._metadata)

    def run(self, *_: Any, **__: Any) -> list[np.ndarray]:
        assert self._run_result is not None
        return [self._run_result]


def test_a_conforming_custom_yolo_session_validates_and_returns_the_input_name() -> None:
    name = _validate_session(_FakeSession(), _entry())  # type: ignore[arg-type]
    assert name == "images"


def test_the_channel_count_must_be_five_plus_class_count() -> None:
    session = _FakeSession(output_shape=[1, 6, TOTAL_ANCHOR_POSITIONS])
    with pytest.raises(ModelContractError, match=r"is not \(1, 7, anchors\)"):
        _validate_session(session, _entry())  # type: ignore[arg-type]


def test_the_anchor_position_count_must_match_three_anchors_per_scale() -> None:
    session = _FakeSession(output_shape=[1, 7, TOTAL_ANCHOR_POSITIONS // 3])
    with pytest.raises(ModelContractError, match="anchor positions"):
        _validate_session(session, _entry())  # type: ignore[arg-type]


def test_metadata_checks_still_apply_to_the_custom_layout() -> None:
    session = _FakeSession(metadata={})
    with pytest.raises(ModelContractError, match="carries no 'names' metadata"):
        _validate_session(session, _entry())  # type: ignore[arg-type]


def test_infer_dispatches_to_the_custom_yolo_decoder(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = np.zeros((1, 7, TOTAL_ANCHOR_POSITIONS), dtype=np.float32)
    raw[0, 4, :] = -20.0  # objectness off everywhere except one planted anchor
    raw[0, 4, 0] = 20.0
    raw[0, 5, 0] = 20.0  # class 0
    # tx=ty=tw=th default 0 -> a small box at the very first grid cell/anchor.

    session = _FakeSession(run_result=raw)
    provider = OnnxProvider(
        session,  # type: ignore[arg-type]
        "images",
        _entry(),
        score_threshold=0.5,
        iou_threshold=0.45,
    )

    found = provider.infer(np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8))

    assert len(found) == 1
    assert found[0].class_name == "dark_gill"
