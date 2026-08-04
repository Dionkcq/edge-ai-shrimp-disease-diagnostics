"""Callable tools exposed to the local assistant.

The assistant never imports the detector directly. It receives a registry of tool
schemas and dispatches a model-requested tool name through that registry. This is
the same seam that can later be backed by MCP without changing the chat route.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import anyio.to_thread

from shrimp_screening.api.dependencies import AppResources
from shrimp_screening.detection.protocol import Detection
from shrimp_screening.imaging.intake import decode_image
from shrimp_screening.imaging.quality import assess_quality
from shrimp_screening.policy.decision import decide

ToolHandler = Callable[[dict[str, Any], bytes | None], Awaitable[dict[str, Any]]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: ToolHandler

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self, tools: tuple[ToolSpec, ...]) -> None:
        self._tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def invoke(
        self, name: str, arguments: dict[str, Any], image: bytes | None
    ) -> dict[str, Any]:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"unknown tool requested: {name}")
        return await tool.handler(arguments, image)


async def screen_image_tool(
    resources: AppResources, _arguments: dict[str, Any], image: bytes | None
) -> dict[str, Any]:
    if image is None:
        return {"status": "needs_image", "message": "Ask the user to upload a shrimp image."}
    decoded = decode_image(image, max_bytes=resources.settings.max_upload_bytes)
    quality = assess_quality(decoded.array, resources.quality_policy)
    if quality.status.value != "PASS" or not resources.detector.metadata.available:
        return {
            "status": "abstained",
            "decision": "UNABLE_TO_ASSESS",
            "quality_status": quality.status.value,
            "quality_reasons": [reason.value for reason in quality.reasons],
            "model_available": resources.detector.metadata.available,
        }
    await asyncio.wait_for(
        resources.inference_gate.acquire(), timeout=resources.settings.queue_wait_timeout_seconds
    )
    try:
        detections: list[Detection] = await anyio.to_thread.run_sync(
            resources.detector.infer, decoded.array
        )
    finally:
        resources.inference_gate.release()
    outcome = decide(
        detections,
        quality.status,
        model_available=resources.detector.metadata.available,
        policy=resources.decision_policy,
    )
    return {
        "status": "screened",
        "request_id": str(uuid.uuid4()),
        "decision": outcome.decision.value,
        "confidence_band": outcome.confidence_band.value,
        "marker_count": len(outcome.markers),
        "markers": [
            {
                "class_name": item.detection.class_name,
                "score": item.detection.score,
                "box": item.detection.box,
            }
            for item in outcome.markers
        ],
        "model_id": resources.detector.metadata.model_id,
        "provider": resources.detector.metadata.provider.value,
    }


def build_tool_registry(resources: AppResources) -> ToolRegistry:
    return ToolRegistry(
        (
            ToolSpec(
                name="screen_shrimp_image",
                description=(
                    "Run the configured CNN detector on the currently uploaded shrimp image. "
                    "Use this whenever the user asks to screen, inspect, or interpret the image."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string", "description": "Why screening is needed."}
                    },
                    "required": ["reason"],
                },
                handler=lambda arguments, image: screen_image_tool(resources, arguments, image),
            ),
        )
    )
