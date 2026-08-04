"""A thin async client for one local Ollama server's ``/api/generate`` endpoint.

This is the only module allowed to know Ollama's wire format. It does not
interpret the model's text in any way -- parsing and safety-scanning the
response is :mod:`shrimp_screening.llm.advisor`'s job -- it only turns a prompt
into a raw string, or raises :class:`OllamaError` for every way that can fail:
the server is not running, the model is not pulled, the request timed out, or
the response body is not the JSON shape Ollama documents.
"""

from __future__ import annotations

from typing import Any

import httpx2 as httpx

from shrimp_screening.settings import DEFAULT_LLM_BASE_URL

_GENERATE_PATH = "/api/generate"


class OllamaError(RuntimeError):
    """The local Ollama server could not be reached or answered unusably."""


class OllamaClient:
    """Calls one model on one local Ollama server."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_LLM_BASE_URL,
        model: str,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base_url = base_url
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._transport = transport

    async def generate(self, *, system: str, prompt: str) -> str:
        """Return the model's raw text response to one non-streamed prompt."""
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            try:
                response = await client.post(
                    _GENERATE_PATH,
                    json={
                        "model": self._model,
                        "system": system,
                        "prompt": prompt,
                        "stream": False,
                        # Low temperature: this text must stay close to the cited
                        # guidance it is grounded in, not roam creatively.
                        "options": {"temperature": 0.2},
                    },
                )
                response.raise_for_status()
            except httpx.RequestError as exc:
                raise OllamaError(
                    f"could not reach the local Ollama server at {self._base_url!r}; is it running?"
                ) from exc
            except httpx.HTTPStatusError as exc:
                raise OllamaError(
                    f"Ollama returned HTTP {exc.response.status_code} for model "
                    f"{self._model!r}; is it pulled?"
                ) from exc

            try:
                payload: Any = response.json()
            except ValueError as exc:
                raise OllamaError("Ollama returned a response that was not valid JSON") from exc

        text = payload.get("response") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            raise OllamaError("Ollama returned no text in its response")
        return text

    async def chat(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Call Ollama's chat API, including function-tool definitions."""
        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout_seconds,
            transport=self._transport,
        ) as client:
            try:
                response = await client.post(
                    "/api/chat",
                    json={
                        "model": self._model,
                        "messages": messages,
                        "tools": tools,
                        "stream": False,
                        "options": {"temperature": 0.2},
                    },
                )
                response.raise_for_status()
            except httpx.RequestError as exc:
                raise OllamaError("could not reach the local Ollama chat server") from exc
            except httpx.HTTPStatusError as exc:
                raise OllamaError(
                    f"Ollama chat returned HTTP {exc.response.status_code} for model "
                    f"{self._model!r}"
                ) from exc
            try:
                payload: Any = response.json()
            except ValueError as exc:
                raise OllamaError("Ollama chat returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("message"), dict):
            raise OllamaError("Ollama chat returned no message")
        return payload
