"""Process-wide state, and the one function that builds it.

Everything that can fail is resolved here, before the first request: policy files
are parsed and hashed, the guidance corpus is validated against the lexicon, and
the detector is built. A misconfiguration is a process that refuses to start,
which is visible; a process that starts and then returns 500s is not.

The one exception is the detector under the ``unavailable`` provider, which
"succeeds" by design and reports that no model is installed. That is the default
state of this repository and the correct answer for a clean checkout.

This module is deliberately free of HTTP: it knows nothing about requests, ASGI or
FastAPI. :mod:`shrimp_server.dependencies` is what attaches the result to an
application and hands it to a route.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from shrimp_screening.ai.memory import ConversationMemory
from shrimp_screening.detection.protocol import MarkerDetector
from shrimp_screening.detection.providers import build_detector
from shrimp_screening.guidance.store import GuidanceCorpus, load_guidance
from shrimp_screening.llm.client import OllamaClient
from shrimp_screening.policy.loader import (
    DecisionPolicy,
    QualityPolicy,
    load_decision_policy,
    load_quality_policy,
)
from shrimp_screening.settings import Settings, load_settings


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
    #: ``None`` unless ``settings.llm_enabled`` -- absence, not a flag to check
    #: separately, is what the advice route treats as "feature not available".
    llm_client: OllamaClient | None
    chat_memory: ConversationMemory


def _build_llm_client(settings: Settings) -> OllamaClient | None:
    if not settings.llm_enabled:
        return None
    return OllamaClient(
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )


def build_resources(
    settings: Settings | None = None,
    *,
    detector: MarkerDetector | None = None,
    llm_client: OllamaClient | None = None,
) -> AppResources:
    """Load and validate every piece of process-wide state.

    ``detector`` and ``llm_client`` are injectable so a test can exercise a
    provider, or a fake local model, without reaching through an environment
    variable or a real network call. Nothing else about the wiring changes.
    """
    resolved = settings if settings is not None else load_settings()
    quality_policy = load_quality_policy()
    decision_policy = load_decision_policy()
    guidance = load_guidance()
    return AppResources(
        settings=resolved,
        quality_policy=quality_policy,
        decision_policy=decision_policy,
        detector=detector if detector is not None else build_detector(resolved, decision_policy),
        guidance=guidance,
        inference_gate=asyncio.Semaphore(resolved.max_concurrent_inferences),
        llm_client=llm_client if llm_client is not None else _build_llm_client(resolved),
        chat_memory=ConversationMemory(),
    )
