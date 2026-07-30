"""Everything the routes need, resolved once at application startup.

Policy files, the guidance corpus and the detector are read and validated while
the process is starting. A malformed policy or an uncited guidance item is
therefore a startup failure, not a 500 discovered by the first user.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import cast

from starlette.requests import Request

from shrimp_screening.detection.protocol import MarkerDetector
from shrimp_screening.guidance.store import GuidanceCorpus
from shrimp_screening.policy.loader import DecisionPolicy, QualityPolicy
from shrimp_screening.settings import Settings

#: Key under which the resource bundle is stored on the ASGI app state.
RESOURCES_ATTRIBUTE = "resources"


@dataclass(frozen=True, slots=True)
class AppResources:
    """Immutable process-wide state."""

    settings: Settings
    quality_policy: QualityPolicy
    decision_policy: DecisionPolicy
    detector: MarkerDetector
    guidance: GuidanceCorpus
    #: Bounds concurrent inference. One ORT session on two physical cores is
    #: faster than two competing ones, and an unbounded queue turns a slow request
    #: into a browser tab that never resolves.
    inference_gate: asyncio.Semaphore


def get_resources(request: Request) -> AppResources:
    return cast(AppResources, getattr(request.app.state, RESOURCES_ATTRIBUTE))


def get_request_id(request: Request) -> str:
    return cast(str, request.state.request_id)
