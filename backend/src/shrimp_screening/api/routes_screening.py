"""``POST /api/v1/screenings`` -- the one endpoint that does the work.

The pipeline, in order, is: bounded read, decode, quality gate, inference,
decision, response. Each stage can end the request, and each way of ending it is
an explicit, named outcome rather than an exception that leaks upward.

Two things are worth calling out because they look like extra work:

**Inference runs in a worker thread behind a semaphore.** onnxruntime is
synchronous and CPU-bound; calling it on the event loop would block every other
request including ``/livez`` for the duration. The semaphore bounds it to one at a
time, and waiting for the semaphore is bounded by a timeout that yields
``503 SERVICE_BUSY`` with ``Retry-After`` rather than an unbounded queue.

**An inference failure is an abstention, not a 500.** If the detector raises, the
system still knows the photograph was valid and its quality acceptable, so the
useful answer is ``UNABLE_TO_ASSESS`` / ``INFERENCE_FAILED`` rather than an error
page that tells the user nothing.
"""

from __future__ import annotations

import asyncio
import logging
from time import perf_counter
from typing import Annotated

import anyio.to_thread
import numpy as np
from fastapi import APIRouter, Depends, Query
from starlette.requests import Request

from shrimp_screening.api.dependencies import AppResources, get_request_id, get_resources
from shrimp_screening.api.errors import ApiProblemError
from shrimp_screening.contracts.enums import (
    DatasetMappingStatus,
    NoticeCode,
    ProblemCode,
    ProviderKind,
    QualityStatus,
)
from shrimp_screening.contracts.screening import (
    BoundingBox,
    MarkerObservation,
    ModelInfo,
    ScreeningResult,
    Timings,
)
from shrimp_screening.detection.fixture_provider import FixtureProvider, UnknownScenarioError
from shrimp_screening.detection.protocol import Detection, MarkerDetector
from shrimp_screening.imaging.intake import decode_image
from shrimp_screening.imaging.quality import assess_quality
from shrimp_screening.imaging.upload import read_single_image_part
from shrimp_screening.limitations import limitation_ids_for
from shrimp_screening.policy.decision import decide

router = APIRouter(prefix="/api/v1", tags=["screening"])

Resources = Annotated[AppResources, Depends(get_resources)]
RequestId = Annotated[str, Depends(get_request_id)]

#: The multipart field name. Fixed, and the only part that is read.
IMAGE_FIELD = "image"

_logger = logging.getLogger("shrimp_screening.screening")


def _select_detector(resources: AppResources, scenario: str | None) -> MarkerDetector:
    """Apply an explicit demonstration scenario, if one is both given and legal."""
    if scenario is None:
        return resources.detector
    if not isinstance(resources.detector, FixtureProvider):
        raise ApiProblemError(
            ProblemCode.MALFORMED_REQUEST,
            400,
            "The 'scenario' parameter selects canned demonstration data and is only "
            "accepted when the service is running on the fixture provider.",
        )
    try:
        return resources.detector.with_scenario(scenario)
    except UnknownScenarioError as exc:
        raise ApiProblemError(
            ProblemCode.MALFORMED_REQUEST, 400, "Unknown demonstration scenario."
        ) from exc


async def _run_inference(
    resources: AppResources, detector: MarkerDetector, image: np.ndarray
) -> tuple[list[Detection], bool]:
    """Return detections and whether inference failed, never raising for the latter."""
    settings = resources.settings
    try:
        await asyncio.wait_for(
            resources.inference_gate.acquire(), timeout=settings.queue_wait_timeout_seconds
        )
    except TimeoutError as exc:
        raise ApiProblemError(
            ProblemCode.SERVICE_BUSY,
            503,
            "The service is already screening another photograph. Try again shortly.",
            retry_after_seconds=settings.retry_after_seconds,
        ) from exc
    try:
        return await anyio.to_thread.run_sync(detector.infer, image), False
    except Exception:
        # The photograph and its quality are still known; abstaining is more useful
        # than a 500. The traceback goes to the server log, keyed by request id.
        _logger.exception("inference failed")
        return [], True
    finally:
        resources.inference_gate.release()


def _notices(resources: AppResources, detector: MarkerDetector) -> list[NoticeCode]:
    metadata = detector.metadata
    notices: list[NoticeCode] = []
    if metadata.provider is ProviderKind.UNAVAILABLE:
        notices.append(NoticeCode.MODEL_NOT_INSTALLED)
    if metadata.demonstration:
        notices.append(NoticeCode.DEMONSTRATION_DATA_NOT_A_REAL_RESULT)
    if metadata.mapping_status is DatasetMappingStatus.PROVISIONAL_UNCONFIRMED:
        notices.append(NoticeCode.DATASET_CLASS_MAPPING_UNCONFIRMED)
    if any(
        "UNCALIBRATED" in status.upper()
        for status in (resources.decision_policy.status, resources.quality_policy.status)
    ):
        notices.append(NoticeCode.THRESHOLDS_UNCALIBRATED)
    return notices


@router.post("/screenings", response_model=ScreeningResult)
async def create_screening(
    request: Request,
    resources: Resources,
    request_id: RequestId,
    scenario: Annotated[
        str | None,
        Query(
            description="Fixture-provider demonstration scenario. Rejected on any other provider.",
            max_length=32,
        ),
    ] = None,
) -> ScreeningResult:
    started = perf_counter()
    detector = _select_detector(resources, scenario)

    payload = await read_single_image_part(
        request, IMAGE_FIELD, resources.settings.max_upload_bytes
    )
    decoded = decode_image(payload, max_bytes=resources.settings.max_upload_bytes)
    # Drop the reference to the encoded bytes as soon as the pixels exist; nothing
    # downstream needs the original file and nothing writes it anywhere.
    del payload
    after_intake = perf_counter()

    quality = assess_quality(decoded.array, resources.quality_policy)
    after_quality = perf_counter()

    detections: list[Detection] = []
    inference_failed = False
    # Stays exactly 0.0 when inference does not run. Timing two adjacent
    # perf_counter() calls would report a fraction of a microsecond of "inference"
    # on a request where no model was ever consulted.
    inference_ms = 0.0
    if quality.status is QualityStatus.PASS and detector.metadata.available:
        inference_started = perf_counter()
        detections, inference_failed = await _run_inference(resources, detector, decoded.array)
        inference_ms = (perf_counter() - inference_started) * 1000.0

    outcome = decide(
        detections,
        quality.status,
        model_available=detector.metadata.available,
        policy=resources.decision_policy,
        inference_failed=inference_failed,
    )

    metadata = detector.metadata
    return ScreeningResult(
        request_id=request_id,
        decision=outcome.decision,
        abstention_reason=outcome.abstention_reason,
        quality=quality,
        markers=[
            MarkerObservation(
                class_index=marker.detection.class_index,
                class_name=marker.detection.class_name,
                role=marker.role,
                score=marker.detection.score,
                box=BoundingBox(
                    x1=marker.detection.box[0],
                    y1=marker.detection.box[1],
                    x2=marker.detection.box[2],
                    y2=marker.detection.box[3],
                ),
            )
            for marker in outcome.markers
        ],
        confidence_band=outcome.confidence_band,
        image=decoded.info,
        model=ModelInfo(
            available=metadata.available,
            provider=metadata.provider,
            model_id=metadata.model_id,
            version=metadata.version,
            output_layout=metadata.output_layout,
            class_names={str(index): name for index, name in metadata.class_names.items()},
            dataset_mapping_status=metadata.mapping_status,
            is_demonstration_data=metadata.demonstration,
        ),
        notices=_notices(resources, detector),
        guidance_refs=[resources.guidance.for_decision(outcome.decision).guidance_id],
        limitations=limitation_ids_for(outcome.decision),
        timings_ms=Timings(
            intake_ms=(after_intake - started) * 1000.0,
            quality_ms=(after_quality - after_intake) * 1000.0,
            inference_ms=inference_ms,
            total_ms=(perf_counter() - started) * 1000.0,
        ),
    )
