"""`data/model_registry.json` is an allowlist, so every defect in it must be a refusal.

The registry is the only thing standing between "somebody dropped a file into
`models/`" and "the service executed it". A parser that shrugs at a malformed entry,
or that accepts a truncated digest, turns the allowlist into a suggestion -- so each
test below is a way the file can be wrong, and each one must raise.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from shrimp_screening.contracts.enums import DatasetMappingStatus, OutputLayout
from shrimp_screening.detection.registry import (
    ModelRegistry,
    RegisteredModel,
    RegistryError,
    load_registry,
    sha256_of,
)
from shrimp_screening.paths import data_dir
from tests.support.onnx_factory import registry_document, write_detect_model

VALID_ENTRY: dict[str, Any] = {
    "model_id": "shrimp-marker-detect",
    "version": "1.0.0",
    "filename": "shrimp-marker-detect-v1.onnx",
    "sha256": "a" * 64,
    "input_size": 640,
    "class_names": {"0": "dark_gill", "1": "white_spot"},
    "opset": 17,
    "output_layout": "ultralytics_v8_detect_v1",
    "anchors": [],
    "dataset_mapping_status": "PROVISIONAL_UNCONFIRMED",
    "artifact_license": "AGPL-3.0-or-later",
    "training_toolchain": "ultralytics 8.3",
}


def _write(path: Path, document: object) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _load_entry(tmp_path: Path, **overrides: object) -> RegisteredModel:
    entry = VALID_ENTRY | overrides
    return load_registry(_write(tmp_path / "registry.json", {"models": [entry]})).models[0]


def _expect_error(tmp_path: Path, document: object, message: str) -> None:
    with pytest.raises(RegistryError, match=message):
        load_registry(_write(tmp_path / "registry.json", document))


# ---------------------------------------------------------------------------
# The state of this repository: an empty allowlist that refuses everything.
# ---------------------------------------------------------------------------


def test_the_committed_registry_parses_and_vouches_for_nothing() -> None:
    """A clean checkout must refuse every ONNX load, and this is why."""
    registry = load_registry()
    assert registry.models == ()
    assert registry.is_empty is True


def test_the_committed_registry_is_the_default_lookup_path() -> None:
    """`load_registry()` with no argument must read the reviewed file, not a guess."""
    expected = data_dir() / "model_registry.json"
    assert expected.is_file()
    assert load_registry() == load_registry(expected)


# ---------------------------------------------------------------------------
# Lookups.
# ---------------------------------------------------------------------------


def test_a_well_formed_entry_parses_into_typed_values(tmp_path: Path) -> None:
    entry = _load_entry(tmp_path)
    assert entry.model_id == "shrimp-marker-detect"
    assert entry.input_size == 640
    assert entry.class_names == {0: "dark_gill", 1: "white_spot"}
    assert entry.output_layout is OutputLayout.ULTRALYTICS_V8_DETECT_V1
    assert entry.dataset_mapping_status is DatasetMappingStatus.PROVISIONAL_UNCONFIRMED
    assert entry.opset == 17


def test_lookup_by_digest_finds_the_entry_and_by_a_stranger_raises(tmp_path: Path) -> None:
    registry = load_registry(_write(tmp_path / "registry.json", {"models": [VALID_ENTRY]}))
    assert registry.by_sha256("a" * 64).model_id == "shrimp-marker-detect"
    with pytest.raises(RegistryError, match="refusing to load an unvouched artifact"):
        registry.by_sha256("b" * 64)


def test_lookup_by_model_id_finds_the_entry_and_by_a_stranger_raises(tmp_path: Path) -> None:
    registry = load_registry(_write(tmp_path / "registry.json", {"models": [VALID_ENTRY]}))
    assert registry.by_model_id("shrimp-marker-detect").sha256 == "a" * 64
    with pytest.raises(RegistryError, match="no model with id 'ghost' is registered"):
        registry.by_model_id("ghost")


def test_the_digest_lookup_scans_past_a_non_matching_entry(tmp_path: Path) -> None:
    """Two entries, and the wanted one second: the loop must not stop at the first."""
    other = VALID_ENTRY | {
        "model_id": "other",
        "filename": "other.onnx",
        "sha256": "b" * 64,
    }
    registry = load_registry(_write(tmp_path / "registry.json", {"models": [other, VALID_ENTRY]}))
    assert registry.by_sha256("a" * 64).model_id == "shrimp-marker-detect"
    assert registry.by_model_id("shrimp-marker-detect").sha256 == "a" * 64
    assert registry.is_empty is False


# ---------------------------------------------------------------------------
# Document-level defects.
# ---------------------------------------------------------------------------


def test_a_missing_registry_is_an_error_not_an_empty_allowlist(tmp_path: Path) -> None:
    """Silently reading "no models" out of a missing file would hide a broken deploy."""
    with pytest.raises(RegistryError, match="could not be read"):
        load_registry(tmp_path / "absent.json")


def test_a_directory_where_the_registry_should_be_is_an_error(tmp_path: Path) -> None:
    (tmp_path / "registry.json").mkdir()
    with pytest.raises(RegistryError, match="could not be read"):
        load_registry(tmp_path / "registry.json")


def test_invalid_json_is_an_error(tmp_path: Path) -> None:
    path = tmp_path / "registry.json"
    path.write_text('{"models": [', encoding="utf-8")
    with pytest.raises(RegistryError, match="not valid JSON"):
        load_registry(path)


@pytest.mark.parametrize(
    "document",
    [
        pytest.param([], id="array"),
        pytest.param("models", id="string"),
        pytest.param({}, id="no-models-key"),
        pytest.param({"models": {}}, id="models-is-object"),
        pytest.param({"models": None}, id="models-is-null"),
    ],
)
def test_a_document_that_is_not_an_object_with_a_models_array_is_an_error(
    tmp_path: Path, document: object
) -> None:
    _expect_error(tmp_path, document, "must be an object with a 'models' array")


@pytest.mark.parametrize("entry", [pytest.param("x", id="string"), pytest.param(7, id="int")])
def test_an_entry_that_is_not_an_object_is_an_error(tmp_path: Path, entry: object) -> None:
    _expect_error(tmp_path, {"models": [entry]}, "each registry entry must be an object")


@pytest.mark.parametrize("field", sorted(VALID_ENTRY))
def test_every_field_is_required(tmp_path: Path, field: str) -> None:
    """No field here is optional: each one is part of what was reviewed."""
    incomplete = {key: value for key, value in VALID_ENTRY.items() if key != field}
    _expect_error(tmp_path, {"models": [incomplete]}, f"missing required field '{field}'")


# ---------------------------------------------------------------------------
# The digest. A truncated or non-hex digest is what an accidental paste looks like.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "digest",
    [
        pytest.param("a" * 63, id="too-short"),
        pytest.param("a" * 65, id="too-long"),
        pytest.param("", id="empty"),
        pytest.param("g" * 64, id="non-hex"),
        pytest.param("sha256:" + "a" * 64, id="prefixed"),
        pytest.param("a" * 32, id="md5-length"),
    ],
)
def test_a_digest_that_is_not_64_hex_characters_is_an_error(tmp_path: Path, digest: str) -> None:
    _expect_error(
        tmp_path, {"models": [VALID_ENTRY | {"sha256": digest}]}, "64-character hex digest"
    )


def test_an_uppercase_digest_is_normalized_so_it_still_matches(tmp_path: Path) -> None:
    """`sha256_of` returns lowercase; a digest pasted from another tool may not be."""
    entry = _load_entry(tmp_path, sha256="ABCDEF" + "0" * 58)
    assert entry.sha256 == "abcdef" + "0" * 58


# ---------------------------------------------------------------------------
# `class_names`: the map that decides what every box is called.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("class_names", "message"),
    [
        pytest.param({}, "non-empty object", id="empty"),
        pytest.param([], "non-empty object", id="array"),
        pytest.param("dark_gill", "non-empty object", id="string"),
        pytest.param({"zero": "dark_gill"}, "is not an integer index", id="non-numeric-key"),
        pytest.param({"0": 1}, "must be a non-empty string", id="int-value"),
        pytest.param({"0": ""}, "must be a non-empty string", id="empty-value"),
        pytest.param({"0": None}, "must be a non-empty string", id="null-value"),
        pytest.param({"1": "white_spot"}, r"cover 0\.\.n-1", id="starts-at-one"),
        pytest.param({"0": "a", "2": "b"}, r"cover 0\.\.n-1", id="gapped"),
    ],
)
def test_an_unusable_class_name_map_is_an_error(
    tmp_path: Path, class_names: object, message: str
) -> None:
    _expect_error(tmp_path, {"models": [VALID_ENTRY | {"class_names": class_names}]}, message)


def test_a_single_class_model_is_accepted(tmp_path: Path) -> None:
    """One class covers 0..0, so a single-marker model is legal."""
    assert _load_entry(tmp_path, class_names={"0": "white_spot"}).class_names == {0: "white_spot"}


# ---------------------------------------------------------------------------
# Values that must coerce, and closed vocabularies that must not be widened.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("opset", "seventeen", id="opset"),
        pytest.param("output_layout", "yolov5_transposed", id="output_layout"),
        pytest.param("dataset_mapping_status", "PROBABLY_FINE", id="mapping_status"),
    ],
)
def test_a_value_outside_its_vocabulary_is_an_error(
    tmp_path: Path, field: str, value: object
) -> None:
    _expect_error(tmp_path, {"models": [VALID_ENTRY | {field: value}]}, "is invalid")


@pytest.mark.parametrize("value", ["640", 640.9, True, 650, 0])
def test_input_size_must_be_a_positive_integer_multiple_of_32(
    tmp_path: Path, value: object
) -> None:
    _expect_error(tmp_path, {"models": [VALID_ENTRY | {"input_size": value}]}, "input_size")


# ---------------------------------------------------------------------------
# The digest function itself.
# ---------------------------------------------------------------------------


def test_the_digest_matches_hashlib_over_the_same_bytes(tmp_path: Path) -> None:
    payload = b"synthetic artifact bytes" * 100
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    assert sha256_of(path) == hashlib.sha256(payload).hexdigest()


def test_the_digest_is_chunk_size_independent(tmp_path: Path) -> None:
    """A large artifact is streamed, so the chunking must not change the answer."""
    payload = bytes(range(256)) * 900
    path = tmp_path / "artifact.bin"
    path.write_bytes(payload)
    reference = hashlib.sha256(payload).hexdigest()
    assert sha256_of(path, chunk_size=1) == reference
    assert sha256_of(path, chunk_size=7) == reference
    assert sha256_of(path) == reference


def test_the_digest_of_an_empty_file_is_the_empty_digest(tmp_path: Path) -> None:
    path = tmp_path / "empty.bin"
    path.write_bytes(b"")
    assert sha256_of(path) == hashlib.sha256(b"").hexdigest()


def test_one_changed_byte_changes_the_digest(tmp_path: Path) -> None:
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"\x00" * 4096)
    second.write_bytes(b"\x00" * 4095 + b"\x01")
    assert sha256_of(first) != sha256_of(second)


# ---------------------------------------------------------------------------
# End to end: a real artifact, a real digest, a real file.
# ---------------------------------------------------------------------------


def test_a_registry_written_for_a_real_artifact_resolves_it(tmp_path: Path) -> None:
    artifact = write_detect_model(tmp_path / "synthetic-detect.onnx")
    registry = load_registry(_write(tmp_path / "registry.json", registry_document(artifact)))
    entry = registry.by_sha256(sha256_of(artifact))
    assert entry.filename == artifact.name
    assert entry.input_size == 64


def test_an_empty_models_array_is_valid_and_empty(tmp_path: Path) -> None:
    registry = load_registry(_write(tmp_path / "registry.json", {"models": []}))
    assert registry == ModelRegistry(models=())


@pytest.mark.parametrize(
    ("field", "second_value"),
    [
        ("model_id", "shrimp-marker-detect"),
        ("filename", "shrimp-marker-detect-v1.onnx"),
        ("sha256", "a" * 64),
    ],
)
def test_registry_rejects_duplicate_identity_fields(
    tmp_path: Path, field: str, second_value: str
) -> None:
    second = VALID_ENTRY | {
        "model_id": "other-model",
        "filename": "other.onnx",
        "sha256": "b" * 64,
        field: second_value,
    }
    _expect_error(tmp_path, {"models": [VALID_ENTRY, second]}, f"duplicate {field}")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("artifact_license", "MIT", "artifact_license"),
        ("training_toolchain", "custom trainer", "training_toolchain"),
        ("opset", 99, "opset"),
        ("model_id", "", "model_id"),
    ],
)
def test_registry_rejects_inconsistent_or_unbounded_model_metadata(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    _expect_error(tmp_path, {"models": [VALID_ENTRY | {field: value}]}, message)


def test_registry_rejects_unknown_entry_fields(tmp_path: Path) -> None:
    _expect_error(
        tmp_path,
        {"models": [VALID_ENTRY | {"unreviewed": True}]},
        "unknown fields",
    )


# ---------------------------------------------------------------------------
# `anchors`: only meaningful for the anchor-based custom head.
# ---------------------------------------------------------------------------

CUSTOM_ANCHOR_ENTRY: dict[str, Any] = VALID_ENTRY | {
    "model_id": "shrimp-marker-custom",
    "filename": "shrimp-marker-custom-v1.onnx",
    "sha256": "c" * 64,
    "output_layout": "custom_yolo_anchor_v1",
    "anchors": [[10.0, 10.0]] * 9,
    "training_toolchain": "custom-pytorch-yolo torch 2.5.1",
}


def test_a_custom_yolo_entry_with_nine_anchors_is_accepted(tmp_path: Path) -> None:
    entry = _load_entry(tmp_path, **CUSTOM_ANCHOR_ENTRY)
    assert entry.output_layout is OutputLayout.CUSTOM_YOLO_ANCHOR_V1
    assert entry.anchors == ((10.0, 10.0),) * 9


@pytest.mark.parametrize(
    "anchors",
    [
        pytest.param([[10.0, 10.0]] * 8, id="too-few"),
        pytest.param([[10.0, 10.0]] * 10, id="too-many"),
        pytest.param([], id="empty"),
    ],
)
def test_a_custom_yolo_entry_without_exactly_nine_anchors_is_refused(
    tmp_path: Path, anchors: object
) -> None:
    _expect_error(
        tmp_path, {"models": [CUSTOM_ANCHOR_ENTRY | {"anchors": anchors}]}, "requires exactly 9"
    )


def test_a_non_empty_anchors_list_on_the_ultralytics_layout_is_refused(tmp_path: Path) -> None:
    _expect_error(
        tmp_path,
        {"models": [VALID_ENTRY | {"anchors": [[10.0, 10.0]]}]},
        "does not use anchors",
    )


@pytest.mark.parametrize(
    "anchors",
    [
        pytest.param("not-a-list", id="not-a-list"),
        pytest.param([[10.0]], id="wrong-arity"),
        pytest.param([[10.0, -1.0]] * 9, id="non-positive"),
        pytest.param([[10.0, "x"]] * 9, id="non-numeric"),
        pytest.param([[10.0, True]] * 9, id="bool-not-number"),
    ],
)
def test_a_malformed_anchors_entry_is_refused(tmp_path: Path, anchors: object) -> None:
    _expect_error(
        tmp_path, {"models": [CUSTOM_ANCHOR_ENTRY | {"anchors": anchors}]}, "anchor|numeric pair"
    )
