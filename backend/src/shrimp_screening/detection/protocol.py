"""The boundary between "something that produces boxes" and the rest of the app.

Everything downstream depends on this Protocol, never on onnxruntime, a file
format or a trainer. Swapping the detector is a new class satisfying
:class:`MarkerDetector`; no other module changes.

:class:`Detection` deliberately carries no screening *role*. A detector reports
what its own metadata calls the class; the decision policy decides what that class
means. Keeping those apart is what stops an index-to-label constant from appearing
anywhere in the backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from shrimp_screening.contracts.enums import DatasetMappingStatus, OutputLayout, ProviderKind


@dataclass(frozen=True, slots=True)
class Detection:
    """One retained box in normalized, EXIF-corrected original-image coordinates."""

    class_index: int
    class_name: str
    score: float
    box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class DetectorMetadata:
    """What the API reports about the detector in every single response."""

    available: bool
    provider: ProviderKind
    model_id: str | None
    version: str | None
    output_layout: OutputLayout | None
    class_names: dict[int, str]
    mapping_status: DatasetMappingStatus
    #: True when the numbers are synthetic. The API turns this into a notice that a
    #: user interface is required to render permanently.
    demonstration: bool


class DetectorUnavailableError(RuntimeError):
    """A detector could not be constructed. Fail closed; never fall back silently."""


@runtime_checkable
class MarkerDetector(Protocol):
    """Produces detections for one decoded RGB image."""

    @property
    def metadata(self) -> DetectorMetadata:
        """Facts about this detector, echoed in every response."""
        ...

    def infer(self, image: np.ndarray) -> list[Detection]:
        """Return detections for an ``(h, w, 3)`` uint8 RGB array."""
        ...
