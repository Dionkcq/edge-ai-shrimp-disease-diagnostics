"""The detector that exists when no detector exists.

This is the default on a clean checkout and it is not a degraded mode to be
apologised for -- it is the honest state of a repository that contains no trained
weights. It reports ``available=False``, which drives ``/readyz`` to 503 and every
screening to ``UNABLE_TO_ASSESS`` / ``MODEL_UNAVAILABLE``.

The alternative -- returning an empty detection list from a "working" detector --
would render as ``NO_TARGET_MARKER_DETECTED``: a missing model would look exactly
like a clean shrimp, and the entire test suite would stay vacuously green.
"""

from __future__ import annotations

import numpy as np

from shrimp_screening.contracts.enums import DatasetMappingStatus, ProviderKind
from shrimp_screening.detection.protocol import Detection, DetectorMetadata

_METADATA = DetectorMetadata(
    available=False,
    provider=ProviderKind.UNAVAILABLE,
    model_id=None,
    version=None,
    output_layout=None,
    class_names={},
    mapping_status=DatasetMappingStatus.NOT_APPLICABLE,
    demonstration=False,
)


class UnavailableProvider:
    """Answers "no model is installed" to every question."""

    @property
    def metadata(self) -> DetectorMetadata:
        return _METADATA

    def infer(self, image: np.ndarray) -> list[Detection]:
        del image
        return []
