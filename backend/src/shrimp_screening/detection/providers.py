"""Selecting a detector. There is no fallback path, by design.

Note on the score threshold passed to every provider: it is the policy's
``candidate_detection_score`` (the decode/NMS floor), **not**
``minimum_detection_score`` (the reporting bar). Decoding at the reporting bar
would discard every weak sighting before ``policy.decide`` could see it, which
would make the ``LOW_CONFIDENCE`` abstention unreachable and turn a marginal
detection into a confident ``NO_TARGET_MARKER_DETECTED``.


Each provider is chosen explicitly and fails loudly if it cannot be built:

* ``unavailable`` -- always succeeds, always reports "no model installed";
* ``fixture``     -- always succeeds, always flags itself as synthetic;
* ``onnx``        -- succeeds only against a registered, contract-checked artifact.

The tempting fourth behaviour, "try onnx and fall back to unavailable", is exactly
what would turn a failed deployment into a silently degraded one. Asking for
``onnx`` and getting something else is an error.
"""

from __future__ import annotations

from pathlib import Path

from shrimp_screening.detection.fixture_provider import DEFAULT_SCENARIO, FixtureProvider
from shrimp_screening.detection.onnx_provider import ModelContractError, load_onnx_provider
from shrimp_screening.detection.protocol import DetectorUnavailableError, MarkerDetector
from shrimp_screening.detection.registry import RegistryError, load_registry
from shrimp_screening.detection.unavailable_provider import UnavailableProvider
from shrimp_screening.policy.loader import DecisionPolicy
from shrimp_screening.settings import Settings


def build_detector(
    settings: Settings,
    policy: DecisionPolicy,
    *,
    fixture_scenario: str = DEFAULT_SCENARIO,
) -> MarkerDetector:
    """Construct the configured detector or raise :class:`DetectorUnavailableError`."""
    if settings.provider == "unavailable":
        return UnavailableProvider()

    if settings.provider == "fixture":
        return FixtureProvider(
            score_threshold=policy.candidate_detection_score,
            iou_threshold=policy.iou_threshold,
            max_detections=policy.max_detections,
            scenario=fixture_scenario,
        )

    if settings.onnx_model_path is None:
        raise DetectorUnavailableError(
            "SHRIMP_PROVIDER=onnx requires SHRIMP_ONNX_MODEL_PATH. No weights exist "
            "in this repository, so this is the expected state on a clean checkout."
        )
    try:
        registry = load_registry(
            Path(settings.model_registry_path) if settings.model_registry_path else None
        )
        return load_onnx_provider(
            Path(settings.onnx_model_path),
            registry,
            score_threshold=policy.candidate_detection_score,
            iou_threshold=policy.iou_threshold,
            max_detections=policy.max_detections,
        )
    except (ModelContractError, RegistryError) as exc:
        raise DetectorUnavailableError(str(exc)) from exc
