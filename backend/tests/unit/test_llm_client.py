"""The Ollama client is the one module allowed to know the wire format.

Every case here runs against an in-process ``httpx2.MockTransport`` -- no real
network call is ever made, on principle: this suite must pass on a machine with
no Ollama server installed at all, exactly like every other test in this repo.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx2 as httpx
import pytest

from shrimp_screening.llm.client import OllamaClient, OllamaError

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> OllamaClient:
    return OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:7b-instruct-q4_0",
        timeout_seconds=5.0,
        transport=httpx.MockTransport(handler),
    )


def _generate(handler: Handler) -> str:
    return asyncio.run(_client(handler).generate(system="sys", prompt="prompt"))


def test_generate_returns_the_response_field() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/generate"
        body = json.loads(request.content)
        assert body["model"] == "qwen2.5:7b-instruct-q4_0"
        assert body["system"] == "sys"
        assert body["prompt"] == "prompt"
        assert body["stream"] is False
        return httpx.Response(200, json={"response": "hello"})

    assert _generate(handler) == "hello"


def test_a_connection_failure_becomes_an_ollama_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(OllamaError, match="could not reach"):
        _generate(handler)


def test_a_non_2xx_status_becomes_an_ollama_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model not found"})

    with pytest.raises(OllamaError, match="HTTP 404"):
        _generate(handler)


def test_a_non_json_body_becomes_an_ollama_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    with pytest.raises(OllamaError, match="not valid JSON"):
        _generate(handler)


def test_a_missing_response_field_becomes_an_ollama_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"done": True})

    with pytest.raises(OllamaError, match="no text"):
        _generate(handler)


def test_a_blank_response_field_becomes_an_ollama_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "   "})

    with pytest.raises(OllamaError, match="no text"):
        _generate(handler)
