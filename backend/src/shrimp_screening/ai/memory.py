"""Bounded process-local memory for one demo conversation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from threading import Lock

from shrimp_screening.contracts.chat import ChatMessage


@dataclass
class ConversationMemory:
    max_messages: int = 12
    _sessions: dict[str, deque[ChatMessage]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock)

    def append(self, session_id: str, message: ChatMessage) -> None:
        with self._lock:
            history = self._sessions.setdefault(session_id, deque(maxlen=self.max_messages))
            history.append(message)

    def recent(self, session_id: str) -> list[ChatMessage]:
        with self._lock:
            return list(self._sessions.get(session_id, ()))

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)
