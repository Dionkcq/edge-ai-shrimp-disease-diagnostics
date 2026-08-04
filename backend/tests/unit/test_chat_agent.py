from typing import Any

import pytest

from shrimp_screening.ai.agent import ScreeningAgent
from shrimp_screening.ai.memory import ConversationMemory
from shrimp_screening.ai.tools import ToolRegistry, ToolSpec
from shrimp_screening.contracts.chat import ChatMessage


def _grounded_result(decision: str, *, status: str = "screened") -> dict[str, Any]:
    return {
        "status": status,
        "decision": decision,
        "guidance": {
            "headline": "Shrimp screening result available",
            "body": "Follow the cited local guidance for this screening outcome.",
            "sources": [{"title": "WOAH Aquatic Manual"}],
        },
    }


class FakeClient:
    def __init__(self) -> None:
        self.calls: list[list[dict[str, Any]]] = []
        self._step = 0

    async def chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.calls.append(messages)
        self._step += 1
        if self._step == 1:
            return {
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "screen_shrimp_image",
                                "arguments": {"reason": "the user asked for screening"},
                            }
                        }
                    ],
                }
            }
        return {"message": {"role": "assistant", "content": "The image was screened cautiously."}}


class NoToolFirstClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {"message": {"role": "assistant", "content": "Please upload an image."}}
        return {"message": {"role": "assistant", "content": "The image was screened."}}


class HumanHealthcareReplyClient:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            return {"message": {"role": "assistant", "content": "Please upload an image."}}
        return {
            "message": {
                "role": "assistant",
                "content": (
                    "I don't provide medical advice or treatments. Please have the image "
                    "reviewed by a qualified healthcare professional."
                ),
            }
        }


class EmptyNoToolClient:
    def __init__(self, message: dict[str, Any]) -> None:
        self.message = message

    async def chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {"message": self.message}


@pytest.mark.anyio
async def test_agent_dispatches_model_tool_call_and_keeps_short_memory() -> None:
    invoked: list[dict[str, Any]] = []

    async def screen(arguments: dict[str, Any], image: bytes | None) -> dict[str, Any]:
        invoked.append({"arguments": arguments, "image": image})
        return _grounded_result("NO_TARGET_MARKER_DETECTED")

    registry = ToolRegistry(
        (
            ToolSpec(
                name="screen_shrimp_image",
                description="Screen the uploaded image.",
                parameters={"type": "object"},
                handler=screen,
            ),
        )
    )
    client = FakeClient()
    agent = ScreeningAgent(client, registry, ConversationMemory(max_messages=4))

    reply, session_id, messages, records, result = await agent.run(
        session_id=None, message="Please inspect this", image=b"image-bytes"
    )

    assert session_id
    assert "Follow the cited local guidance" in reply
    assert "The image was screened cautiously" not in reply
    assert invoked == [
        {"arguments": {"reason": "the user asked for screening"}, "image": b"image-bytes"}
    ]
    assert [record.name for record in records] == ["screen_shrimp_image"]
    assert result == _grounded_result("NO_TARGET_MARKER_DETECTED")
    assert [message.role for message in messages] == ["user", "tool", "assistant"]
    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_agent_falls_back_to_screening_when_model_omits_tool_call() -> None:
    invoked: list[bytes | None] = []

    async def screen(arguments: dict[str, Any], image: bytes | None) -> dict[str, Any]:
        invoked.append(image)
        return _grounded_result("UNABLE_TO_ASSESS", status="abstained")

    registry = ToolRegistry(
        (
            ToolSpec(
                name="screen_shrimp_image",
                description="Screen the uploaded image.",
                parameters={"type": "object"},
                handler=screen,
            ),
        )
    )
    client = NoToolFirstClient()
    agent = ScreeningAgent(client, registry)

    reply, _, _, records, result = await agent.run(
        session_id=None, message="What does this image show?", image=b"image-bytes"
    )

    assert "Follow the cited local guidance" in reply
    assert "The image was screened." not in reply
    assert invoked == [b"image-bytes"]
    assert [record.name for record in records] == ["screen_shrimp_image"]
    assert result == _grounded_result("UNABLE_TO_ASSESS", status="abstained")
    assert client.calls == 1


@pytest.mark.anyio
async def test_agent_returns_grounded_guidance_without_second_model_pass() -> None:
    guidance_body = (
        "Retake the photograph under even light and compare several shrimp from the same pond."
    )

    async def screen(arguments: dict[str, Any], image: bytes | None) -> dict[str, Any]:
        return {
            "status": "screened",
            "decision": "GILL_DARKENING_MARKER_DETECTED",
            "guidance": {
                "decision": "GILL_DARKENING_MARKER_DETECTED",
                "headline": "A dark-gill-like region was marked",
                "body": guidance_body,
                "sources": [
                    {
                        "title": "Manual of Diagnostic Tests for Aquatic Animals",
                        "publisher": "World Organisation for Animal Health (WOAH)",
                    }
                ],
            },
        }

    registry = ToolRegistry(
        (
            ToolSpec(
                name="screen_shrimp_image",
                description="Screen the uploaded image.",
                parameters={"type": "object"},
                handler=screen,
            ),
        )
    )
    client = HumanHealthcareReplyClient()
    agent = ScreeningAgent(client, registry)

    reply, _, _, records, result = await agent.run(
        session_id=None, message="How do I treat this?", image=b"image-bytes"
    )

    assert "healthcare professional" not in reply.lower()
    assert "medical advice" not in reply.lower()
    assert "aquatic-animal health professional" in reply.lower()
    assert guidance_body in reply
    assert "Manual of Diagnostic Tests for Aquatic Animals" in reply
    assert [record.name for record in records] == ["screen_shrimp_image"]
    assert result is not None
    assert result["decision"] == "GILL_DARKENING_MARKER_DETECTED"
    assert client.calls == 1


@pytest.mark.anyio
@pytest.mark.parametrize("model_message", ({}, {"content": ""}))
async def test_agent_uses_upload_prompt_when_model_returns_no_content(
    model_message: dict[str, Any],
) -> None:
    client = EmptyNoToolClient(model_message)
    agent = ScreeningAgent(client, ToolRegistry(()))

    reply, _, messages, records, result = await agent.run(
        session_id=None,
        message="What should I do now?",
        image=None,
    )

    assert "upload a shrimp photograph" in reply.lower()
    assert [message.role for message in messages] == ["user", "assistant"]
    assert records == []
    assert result is None


def test_memory_evicts_old_turns_and_can_clear_session() -> None:
    memory = ConversationMemory(max_messages=2)
    memory.append("session", ChatMessage(role="user", content="one"))
    memory.append("session", ChatMessage(role="assistant", content="two"))
    memory.append("session", ChatMessage(role="user", content="three"))

    assert [message.content for message in memory.recent("session")] == ["two", "three"]
    memory.clear("session")
    assert memory.recent("session") == []
