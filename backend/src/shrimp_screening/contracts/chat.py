"""Contracts for the short-lived screening assistant conversation."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["user", "assistant", "tool"]
    content: str = Field(min_length=1)


class ToolCallRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    status: Literal["completed", "failed"]


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    session_id: str = Field(min_length=1)
    reply: str = Field(min_length=1)
    messages: list[ChatMessage]
    tool_calls: list[ToolCallRecord]
    tool_result: dict[str, Any] | None = None
