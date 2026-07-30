"""The load-time contract on an ONNX artifact.

This is the module whose failure mode is not a crash. A transposed tensor, a
class-order flip between training and registration, or an `nms=True` export all
decode into confident boxes that look exactly like a working detector -- so every
assertion below exists because the alternative is a wrong answer nobody notices.

Every model here is a real ONNX graph that onnxruntime genuinely loads and runs,
built into `tmp_path` by `tests.support.onnx_factory`. Nothing is mocked: a stub
session would agree with whatever `_validate_session` happens to do, which is the
one thing these tests must not be allowed to do. No committed weights are involved
and none are produced.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest

from shrimp_screening.contracts.enums import DatasetMappingStatus, OutputLayout, ProviderKind
from shrimp_screening.detection.onnx_provider import (
    INTRA_OP_THREADS,
    ModelContractError,
    OnnxProvider,
    _session_options,
    load_onnx_provider,
)
from shrimp_screening.detection.registry import ModelRegistry, RegistryError
from tests.support.onnx_factory import (
    SCORE_QUANTUM,
    SYNTHETIC_CLASS_NAMES,
    SYNTHETIC_INPUT_SIZE,
    default_metadata,
    plant_detection_image,
    registered_model,
    registry_for,
    write_detect_model,
)

CANDIDATE_SCORE = 0.15
IOU = 0.45


def _load(path: Path, registry: ModelRegistry, **overrides: float | int) -> OnnxProvider:
    return load_onnx_provider(
        path,
        registry,
        score_threshold=float(overrides.get("score_threshold", CANDIDATE_SCORE)),
        iou_threshold=float(overrides.get("iou_threshold", IOU)),
        max_detections=int(overrides.get("max_detections", 300)),
    )


@pytest.fixture
def artifact(tmp_path: Path) -> Path:
    """A synthetic detect model that satisfies every clause of the contract."""
    return write_detect_model(tmp_path / "synthetic-detect.onnx")


# ---------------------------------------------------------------------------
# The path a real artifact takes: registered, validated, warmed, inferring.
# ---------------------------------------------------------------------------


def test_a_conforming_artifact_loads_and_reports_the_registry_as_its_identity(
    artifact: Path,
) -> None:
    """The response's model block is the registry's word, not the file's."""
    provider = _load(artifact, registry_for(artifact))
    metadata = provider.metadata
    assert metadata.available is True
    assert metadata.provider is ProviderKind.ONNX
    assert metadata.model_id == "synthetic-detect"
    assert metadata.version == "0.0.0-synthetic"
    assert metadata.output_layout is OutputLayout.ULTRALYTICS_V8_DETECT_V1
    assert metadata.class_names == SYNTHETIC_CLASS_NAMES
    assert metadata.mapping_status is DatasetMappingStatus.PROVISIONAL_UNCONFIRMED
    # A real model result must never be able to present itself as a demonstration.
    assert metadata.demonstration is False


def test_metadata_class_names_are_a_copy_the_caller_cannot_mutate(artifact: Path) -> None:
    provider = _load(artifact, registry_for(artifact))
    provider.metadata.class_names[99] = "injected"
    assert 99 not in _load(artifact, registry_for(artifact)).metadata.class_names


def test_a_planted_box_survives_the_whole_session_and_decode_path(artifact: Path) -> None:
    """End to end through a real ORT session: letterbox, run, decode, un-letterbox.

    The synthetic graph is a function of its input, so this asserts the *wiring* --
    that the batch handed to the session is the letterboxed image and that the tensor
    it returns is read channels-then-anchors. A stub session could not prove either.
    """
    provider = _load(artifact, registry_for(artifact))
    image = plant_detection_image(box_pixels=(32, 32, 16, 16), class_index=1, score=0.8)

    found = provider.infer(image)

    assert len(found) == 1
    assert found[0].class_name == "white_spot"
    assert found[0].class_index == 1
    assert found[0].score == pytest.approx(0.8, abs=SCORE_QUANTUM)
    # cx=32, cy=32, w=16, h=16 on a 64px frame -> xyxy (24, 24, 40, 40) -> /64.
    assert found[0].box == pytest.approx((0.375, 0.375, 0.625, 0.625), abs=1e-4)


def test_the_class_name_comes_from_the_registry_not_from_a_hardcoded_index(
    tmp_path: Path,
) -> None:
    """Relabel both the artifact and the registry and the answer must follow."""
    swapped = {0: "white_spot", 1: "dark_gill"}
    path = write_detect_model(
        tmp_path / "swapped.onnx", metadata=default_metadata(class_names=swapped)
    )
    provider = _load(path, registry_for(path, class_names=swapped))
    image = plant_detection_image(box_pixels=(32, 32, 16, 16), class_index=1, score=0.8)
    assert provider.infer(image)[0].class_name == "dark_gill"


def test_a_score_below_the_candidate_floor_produces_nothing(artifact: Path) -> None:
    provider = _load(artifact, registry_for(artifact))
    weak = plant_detection_image(box_pixels=(32, 32, 16, 16), class_index=1, score=0.10)
    assert provider.infer(weak) == []


def test_overlapping_same_class_boxes_are_suppressed_through_the_real_session(
    artifact: Path,
) -> None:
    provider = _load(artifact, registry_for(artifact))
    image = plant_detection_image(box_pixels=(32, 32, 20, 20), class_index=1, score=0.9)
    second = plant_detection_image(box_pixels=(33, 33, 20, 20), class_index=1, score=0.7, anchor=1)
    # Both plantings write into disjoint pixels, so one image can carry both anchors.
    combined = np.maximum(image, second)

    found = provider.infer(combined)
    assert len(found) == 1
    assert found[0].score == pytest.approx(0.9, abs=SCORE_QUANTUM)


def test_max_detections_is_carried_into_the_decode(artifact: Path) -> None:
    image = plant_detection_image(box_pixels=(10, 10, 8, 8), class_index=1, score=0.9)
    far = plant_detection_image(box_pixels=(50, 50, 8, 8), class_index=1, score=0.7, anchor=1)
    combined = np.maximum(image, far)

    assert len(_load(artifact, registry_for(artifact)).infer(combined)) == 2
    assert len(_load(artifact, registry_for(artifact), max_detections=1).infer(combined)) == 1


def test_a_non_square_photograph_stays_inside_its_own_frame(artifact: Path) -> None:
    """A phone photograph is never square, so the padding band is always in play.

    The grey fill is a real signal to a real graph, and here it lands in the score
    rows -- which is precisely the case where a box decoded out of the padding could
    be reported as lying outside the photograph. Every coordinate must be clipped
    back into `[0, 1]`, or a client would draw an overlay off the image.
    """
    provider = _load(artifact, registry_for(artifact))
    found = provider.infer(np.zeros((96, 128, 3), dtype=np.uint8))
    for detection in found:
        assert all(0.0 <= corner <= 1.0 for corner in detection.box), detection
        assert detection.box[0] <= detection.box[2]
        assert detection.box[1] <= detection.box[3]


def test_warm_up_is_paid_at_load_time_not_on_the_first_upload(
    artifact: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first run through a fresh session costs 0.5-2 s of graph optimization.

    Deferring it to the first request reads as a hang half way through a
    demonstration, so `load_onnx_provider` must have run it already.
    """
    calls: list[int] = []
    original = OnnxProvider.warm_up

    def recording(self: OnnxProvider) -> None:
        calls.append(1)
        original(self)

    monkeypatch.setattr(OnnxProvider, "warm_up", recording)
    _load(artifact, registry_for(artifact))
    assert calls == [1]


def test_warm_up_runs_a_zero_batch_and_yields_no_detections(artifact: Path) -> None:
    provider = _load(artifact, registry_for(artifact))
    provider.warm_up()
    assert provider.infer(np.zeros((64, 64, 3), dtype=np.uint8)) == []


# ---------------------------------------------------------------------------
# Refusing an artifact nobody vouched for.
# ---------------------------------------------------------------------------


def test_a_missing_artifact_is_refused_before_onnxruntime_is_asked(tmp_path: Path) -> None:
    with pytest.raises(ModelContractError, match="no model artifact exists"):
        _load(tmp_path / "absent.onnx", ModelRegistry(models=()))


def test_a_directory_at_the_model_path_is_not_a_model(tmp_path: Path) -> None:
    directory = tmp_path / "model.onnx"
    directory.mkdir()
    with pytest.raises(ModelContractError, match="no model artifact exists"):
        _load(directory, ModelRegistry(models=()))


def test_an_unregistered_digest_is_refused_rather_than_executed(artifact: Path) -> None:
    """Weights arrive out of band. An unrecorded artifact is exactly the one to refuse."""
    with pytest.raises(RegistryError, match="refusing to load an unvouched artifact"):
        _load(artifact, ModelRegistry(models=()))


def test_an_empty_registry_refuses_every_artifact(artifact: Path) -> None:
    empty = ModelRegistry(models=())
    assert empty.is_empty is True
    with pytest.raises(RegistryError):
        _load(artifact, empty)


def test_a_renamed_artifact_is_refused_even_though_its_digest_is_registered(
    artifact: Path,
) -> None:
    """The digest identifies the bytes; the filename is part of what was reviewed.

    A digest that matches under a different name means somebody moved a file around
    outside the review, which is not the same artifact having been approved.
    """
    registry = registry_for(artifact, filename="something-else.onnx")
    with pytest.raises(ModelContractError, match="refusing a renamed artifact"):
        _load(artifact, registry)


def test_onnxruntime_refusing_the_bytes_becomes_a_contract_error(tmp_path: Path) -> None:
    """ORT raises a wide, unstable set of types; none of them may escape this module."""
    junk = tmp_path / "not-a-model.onnx"
    junk.write_bytes(b"\x08\x0a" + b"definitely not a protobuf graph" * 8)
    registry = registry_for(junk)
    with pytest.raises(ModelContractError, match="onnxruntime refused to load the artifact"):
        _load(junk, registry)


def test_a_truncated_artifact_is_refused(artifact: Path, tmp_path: Path) -> None:
    truncated = tmp_path / "truncated.onnx"
    truncated.write_bytes(artifact.read_bytes()[: artifact.stat().st_size // 2])
    with pytest.raises(ModelContractError, match="onnxruntime refused to load the artifact"):
        _load(truncated, registry_for(truncated))


# ---------------------------------------------------------------------------
# The output-layout contract, one clause per case.
# ---------------------------------------------------------------------------


def test_a_transposed_yolov5_output_is_refused_not_misread(tmp_path: Path) -> None:
    """`(1, anchors, 4 + nc)` decodes into garbage that looks like detections."""
    path = write_detect_model(tmp_path / "transposed.onnx", output_shape=(1, 84, 6))
    with pytest.raises(ModelContractError, match="transposed YOLOv5 layout"):
        _load(path, registry_for(path))


def test_an_nms_true_export_is_refused(tmp_path: Path) -> None:
    """`nms=True` emits `(1, max_det, 6)`: six channels of a different meaning."""
    path = write_detect_model(tmp_path / "nms-true.onnx", output_shape=(1, 300, 6))
    with pytest.raises(ModelContractError, match=r"\(1, max_det, 6\)"):
        _load(path, registry_for(path))


def test_a_rank_two_output_is_refused(tmp_path: Path) -> None:
    path = write_detect_model(tmp_path / "flat.onnx", output_shape=(1, 504))
    with pytest.raises(ModelContractError, match="static rank-3 tensor"):
        _load(path, registry_for(path))


def test_an_anchor_count_that_is_not_a_three_scale_head_is_refused(tmp_path: Path) -> None:
    """`sum((S // stride) ** 2)` over 8/16/32 is what pins the detect head."""
    path = write_detect_model(tmp_path / "wrong-anchors.onnx", output_shape=(1, 6, 100))
    with pytest.raises(ModelContractError, match="a three-scale detect head at 64 emits 84"):
        _load(path, registry_for(path))


def test_a_channel_count_that_disagrees_with_the_registered_classes_is_refused(
    artifact: Path,
) -> None:
    """Three registered classes against a two-class artifact must not load."""
    three = {0: "dark_gill", 1: "white_spot", 2: "invented"}
    with pytest.raises(ModelContractError, match=r"is not \(1, 7, anchors\)"):
        _load(artifact, registry_for(artifact, class_names=three))


def test_a_dynamic_input_shape_is_refused(tmp_path: Path) -> None:
    """A dynamic export would accept a differently-shaped batch and letterbox wrong."""
    path = write_detect_model(tmp_path / "dynamic.onnx", dynamic_batch=True)
    with pytest.raises(ModelContractError, match="pinned static"):
        _load(path, registry_for(path))


def test_an_input_size_that_disagrees_with_the_registry_is_refused(artifact: Path) -> None:
    with pytest.raises(ModelContractError, match=r"pinned static \[1, 3, 320, 320\]"):
        _load(artifact, registry_for(artifact, input_size=320))


def test_a_non_float32_input_is_refused(tmp_path: Path) -> None:
    path = write_detect_model(tmp_path / "double.onnx", double_input=True)
    with pytest.raises(ModelContractError, match="is not float32"):
        _load(path, registry_for(path))


def test_more_than_one_input_is_refused(tmp_path: Path) -> None:
    path = write_detect_model(tmp_path / "two-in.onnx", extra_input=True)
    with pytest.raises(ModelContractError, match="expected exactly one input, found 2"):
        _load(path, registry_for(path))


def test_more_than_one_output_is_refused(tmp_path: Path) -> None:
    path = write_detect_model(tmp_path / "two-out.onnx", extra_output=True)
    with pytest.raises(ModelContractError, match="expected exactly one output, found 2"):
        _load(path, registry_for(path))


# ---------------------------------------------------------------------------
# The metadata contract. The class-name equality check is the load-bearing one.
# ---------------------------------------------------------------------------


def test_a_class_order_flip_between_training_and_registration_is_refused(
    tmp_path: Path,
) -> None:
    """The single most dangerous defect this project can ship.

    A flipped order produces a fully working application that confidently reports
    "white spot" for a darkened gill and vice versa. Nothing downstream can detect
    it, so this equality check is the only thing standing in the way.
    """
    flipped = {0: "white_spot", 1: "dark_gill"}
    path = write_detect_model(
        tmp_path / "flipped.onnx", metadata=default_metadata(class_names=flipped)
    )
    with pytest.raises(ModelContractError, match="would mislabel every detection"):
        _load(path, registry_for(path))  # registry still says 0=dark_gill


def test_an_artifact_carrying_no_names_metadata_is_refused(tmp_path: Path) -> None:
    path = write_detect_model(tmp_path / "nameless.onnx", metadata={})
    with pytest.raises(ModelContractError, match="carries no 'names' metadata"):
        _load(path, registry_for(path))


@pytest.mark.parametrize(
    ("names", "message"),
    [
        pytest.param("this is not a literal", "not parseable", id="unparseable"),
        pytest.param("[0, 1]", "not a non-empty mapping", id="list"),
        pytest.param("{}", "not a non-empty mapping", id="empty"),
        pytest.param("{'0': 'dark_gill'}", r"not \{int: str\}", id="string-keys"),
        pytest.param("{0: 1}", r"not \{int: str\}", id="int-values"),
    ],
)
def test_unusable_names_metadata_is_refused(tmp_path: Path, names: str, message: str) -> None:
    metadata = default_metadata() | {"names": names}
    path = write_detect_model(tmp_path / "bad-names.onnx", metadata=metadata)
    with pytest.raises(ModelContractError, match=message):
        _load(path, registry_for(path))


@pytest.mark.parametrize("task", ["segment", "classify", "pose", ""])
def test_a_non_detect_task_is_refused(tmp_path: Path, task: str) -> None:
    """A segment or pose export has a different output meaning entirely."""
    metadata = default_metadata() | {"task": task}
    path = write_detect_model(tmp_path / "wrong-task.onnx", metadata=metadata)
    with pytest.raises(ModelContractError, match="is not 'detect'"):
        _load(path, registry_for(path))


def test_a_missing_task_is_refused(tmp_path: Path) -> None:
    metadata = {key: value for key, value in default_metadata().items() if key != "task"}
    path = write_detect_model(tmp_path / "no-task.onnx", metadata=metadata)
    with pytest.raises(ModelContractError, match="is not 'detect'"):
        _load(path, registry_for(path))


@pytest.mark.parametrize(
    "imgsz",
    [
        pytest.param("[32, 32]", id="smaller"),
        pytest.param("[64, 32]", id="non-square"),
        pytest.param("not a literal", id="unparseable"),
        pytest.param("[64, 64, 64]", id="rank-3"),
        pytest.param("['64', '64']", id="string-members"),
        pytest.param("64.0", id="float"),
    ],
)
def test_an_imgsz_that_disagrees_with_the_registered_size_is_refused(
    tmp_path: Path, imgsz: str
) -> None:
    metadata = default_metadata() | {"imgsz": imgsz}
    path = write_detect_model(tmp_path / "wrong-imgsz.onnx", metadata=metadata)
    with pytest.raises(ModelContractError, match="does not agree with the registered 64"):
        _load(path, registry_for(path))


def test_a_missing_imgsz_is_refused(tmp_path: Path) -> None:
    metadata = {key: value for key, value in default_metadata().items() if key != "imgsz"}
    path = write_detect_model(tmp_path / "no-imgsz.onnx", metadata=metadata)
    with pytest.raises(ModelContractError, match="does not agree with the registered 64"):
        _load(path, registry_for(path))


def test_the_scalar_imgsz_form_ultralytics_also_writes_is_accepted(tmp_path: Path) -> None:
    """`imgsz` is a bare int on some export paths and a two-list on others."""
    metadata = default_metadata() | {"imgsz": str(SYNTHETIC_INPUT_SIZE)}
    path = write_detect_model(tmp_path / "scalar-imgsz.onnx", metadata=metadata)
    assert _load(path, registry_for(path)).metadata.available is True


# ---------------------------------------------------------------------------
# Session configuration. There is no observable proxy for these, and getting them
# wrong shows up as random latency spikes rather than as a failure.
# ---------------------------------------------------------------------------


def test_the_session_is_pinned_to_two_sequential_threads_without_spinning() -> None:
    """Four threads oversubscribe MLAS on a two-core box and starve the event loop."""
    options = _session_options()
    assert INTRA_OP_THREADS == 2
    assert options.intra_op_num_threads == INTRA_OP_THREADS
    assert options.inter_op_num_threads == 1
    assert options.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
    assert options.graph_optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL


def test_the_registered_model_record_is_immutable(artifact: Path) -> None:
    """A frozen entry is what makes "the registry vouched for this" a stable claim."""
    entry = registered_model(artifact)
    with pytest.raises(AttributeError):
        entry.sha256 = "0" * 64  # type: ignore[misc]
