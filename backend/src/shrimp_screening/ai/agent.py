"""Local tool-calling assistant for the shrimp screening workflow."""

from __future__ import annotations

import json
import uuid
from typing import Any, Protocol

from shrimp_screening.ai.memory import ConversationMemory
from shrimp_screening.ai.replies import guard_chat_reply
from shrimp_screening.ai.tools import ToolRegistry
from shrimp_screening.contracts.chat import ChatMessage, ToolCallRecord
from shrimp_screening.llm.client import OllamaError


class ChatModel(Protocol):
    async def chat(
        self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]: ...


_SYSTEM = (
    "You are Pondside screen, an offline shrimp-image screening assistant. "
    "Every question in this application concerns shrimp aquaculture, never human health. "
    "Do not refer to doctors, patients, medical care, or healthcare professionals. "
    "Be concise and cautious. You may describe visible markers, never claim a diagnosis, "
    "never invent a result, and never recommend medication, chemicals, or dosing. When the "
    "user asks to inspect or treat something in an uploaded image, call screen_shrimp_image. "
    "The backend renders image results from cited local guidance rather than free-form prose. "
    "For escalation, say 'qualified aquatic-animal health professional'."
)


class ScreeningAgent:
    def __init__(
        self,
        client: ChatModel,
        tools: ToolRegistry,
        memory: ConversationMemory | None = None,
    ) -> None:
        self._client = client
        self._tools = tools
        self._memory = memory or ConversationMemory()

    async def run(
        self, *, session_id: str | None, message: str, image: bytes | None
    ) -> tuple[str, str, list[ChatMessage], list[ToolCallRecord], dict[str, Any] | None]:
        active_session = session_id or str(uuid.uuid4())
        user_text = message.strip() or "Please inspect the uploaded shrimp image."
        user_message = ChatMessage(role="user", content=user_text)
        self._memory.append(active_session, user_message)
        tool_records: list[ToolCallRecord] = []
        tool_result: dict[str, Any] | None = None
        image_screened = False
        for _ in range(3):
            image_context = (
                " An image is attached to the current user turn. Use the image screening tool "
                "before answering questions about that image."
                if image is not None and not image_screened
                else ""
            )
            history = [
                {"role": "system", "content": _SYSTEM + image_context},
                *[
                    {"role": item.role, "content": item.content}
                    for item in self._memory.recent(active_session)
                ],
            ]
            response = await self._client.chat(messages=history, tools=self._tools.schemas())
            model_message = response["message"]
            calls = model_message.get("tool_calls", [])
            if not isinstance(calls, list) or not calls:
                if image is not None and not image_screened:
                    tool_result = await self._tools.invoke(
                        "screen_shrimp_image", {"reason": user_text}, image
                    )
                    image_screened = True
                    tool_records.append(
                        ToolCallRecord(name="screen_shrimp_image", status="completed")
                    )
                    self._memory.append(
                        active_session,
                        ChatMessage(role="tool", content=json.dumps(tool_result, sort_keys=True)),
                    )
                    reply = guard_chat_reply("", tool_result, user_message=user_text)
                    assistant_message = ChatMessage(role="assistant", content=reply)
                    self._memory.append(active_session, assistant_message)
                    return (
                        reply,
                        active_session,
                        self._memory.recent(active_session),
                        tool_records,
                        tool_result,
                    )
                raw_reply = str(model_message.get("content", "")).strip()
                reply = guard_chat_reply(raw_reply, tool_result, user_message=message)
                assistant_message = ChatMessage(role="assistant", content=reply)
                self._memory.append(active_session, assistant_message)
                return (
                    reply,
                    active_session,
                    self._memory.recent(active_session),
                    tool_records,
                    tool_result,
                )
            for call in calls:
                function = call.get("function", {}) if isinstance(call, dict) else {}
                name = str(function.get("name", ""))
                arguments = function.get("arguments", {})
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, dict):
                    raise OllamaError("the assistant returned invalid tool arguments")
                tool_result = await self._tools.invoke(name, arguments, image)
                image_screened = image_screened or name == "screen_shrimp_image"
                tool_records.append(ToolCallRecord(name=name, status="completed"))
                self._memory.append(
                    active_session,
                    ChatMessage(role="tool", content=json.dumps(tool_result, sort_keys=True)),
                )
            if image_screened and tool_result is not None:
                reply = guard_chat_reply("", tool_result, user_message=user_text)
                assistant_message = ChatMessage(role="assistant", content=reply)
                self._memory.append(active_session, assistant_message)
                return (
                    reply,
                    active_session,
                    self._memory.recent(active_session),
                    tool_records,
                    tool_result,
                )
        raise OllamaError("the assistant exceeded the tool-call limit")
