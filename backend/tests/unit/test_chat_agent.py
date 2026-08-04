from typing import Any

import pytest

from shrimp_screening.ai.agent import ScreeningAgent
from shrimp_screening.ai.memory import ConversationMemory
from shrimp_screening.ai.tools import ToolRegistry, ToolSpec
from shrimp_screening.contracts.chat import ChatMessage


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


@pytest.mark.anyio
async def test_agent_dispatches_model_tool_call_and_keeps_short_memory() -> None:
    invoked: list[dict[str, Any]] = []

    async def screen(arguments: dict[str, Any], image: bytes | None) -> dict[str, Any]:
        invoked.append({"arguments": arguments, "image": image})
        return {"status": "screened", "decision": "NO_TARGET_MARKER_DETECTED"}

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
    assert reply == "The image was screened cautiously."
    assert invoked == [
        {"arguments": {"reason": "the user asked for screening"}, "image": b"image-bytes"}
    ]
    assert [record.name for record in records] == ["screen_shrimp_image"]
    assert result == {"status": "screened", "decision": "NO_TARGET_MARKER_DETECTED"}
    assert [message.role for message in messages] == ["user", "tool", "assistant"]
    assert len(client.calls) == 2
    assert client.calls[1][-1]["role"] == "tool"


def test_memory_evicts_old_turns_and_can_clear_session() -> None:
    memory = ConversationMemory(max_messages=2)
    memory.append("session", ChatMessage(role="user", content="one"))
    memory.append("session", ChatMessage(role="assistant", content="two"))
    memory.append("session", ChatMessage(role="user", content="three"))

    assert [message.content for message in memory.recent("session")] == ["two", "three"]
    memory.clear("session")
    assert memory.recent("session") == []
