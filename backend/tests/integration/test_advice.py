"""``GET /api/v1/advice/{decision}`` -- optional, off by default, fails explicitly.

No case here reaches a real network: ``llm_client`` is injected with an
``httpx2.MockTransport`` exactly the way ``detector`` is injected with a fixture
provider elsewhere in this suite, so the whole suite still passes on a machine
with no Ollama server installed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx2 as httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from shrimp_screening.contracts.enums import Decision
from shrimp_screening.llm.client import OllamaClient
from shrimp_screening.settings import Settings
from shrimp_server.main import create_app

Handler = Callable[[httpx.Request], httpx.Response]

_GOOD_PAYLOAD = {
    "summary": "A marked region was found; it is not a diagnosis.",
    "immediate_actions": ["Retake the photo in even light.", "Log the pond and date."],
    "prevention_actions": ["Keep consistent stocking density."],
    "additional_considerations": ["Ask a professional before acting further."],
}


def _settings(*, llm_enabled: bool) -> Settings:
    return Settings(
        env="test",
        provider="unavailable",
        onnx_model_path=None,
        max_upload_bytes=2 * 1024 * 1024,
        max_concurrent_inferences=1,
        queue_wait_timeout_seconds=1.0,
        retry_after_seconds=1,
        llm_enabled=llm_enabled,
    )


def _app(tmp_path: Path, *, llm_enabled: bool, handler: Handler | None = None) -> FastAPI:
    llm_client = None
    if llm_enabled:
        assert handler is not None
        llm_client = OllamaClient(
            base_url="http://127.0.0.1:11434",
            model="qwen2.5:7b-instruct-q4_0",
            timeout_seconds=1.0,
            transport=httpx.MockTransport(handler),
        )
    return create_app(
        _settings(llm_enabled=llm_enabled),
        llm_client=llm_client,
        frontend_dir=tmp_path / "no-frontend-build",
    )


def _respond(payload: dict[str, object]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": json.dumps(payload)})

    return handler


def test_the_endpoint_is_not_found_when_the_feature_is_disabled(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, llm_enabled=False)) as client:
        response = client.get("/api/v1/advice/WHITE_SPOT_MARKER_DETECTED")
    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"
    assert "not enabled" in response.json()["detail"]


def test_disabled_is_the_default(tmp_path: Path) -> None:
    app = create_app(_settings(llm_enabled=False), frontend_dir=tmp_path / "no-frontend-build")
    with TestClient(app) as client:
        response = client.get("/api/v1/advice/WHITE_SPOT_MARKER_DETECTED")
    assert response.status_code == 404


def test_meta_declares_the_feature_off_so_a_client_never_offers_it(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, llm_enabled=False)) as client:
        body = client.get("/api/v1/meta").json()
    assert body["advice_available"] is False


def test_meta_declares_the_feature_on_when_a_client_is_wired(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, llm_enabled=True, handler=_respond(_GOOD_PAYLOAD))) as client:
        body = client.get("/api/v1/meta").json()
    assert body["advice_available"] is True


def test_unknown_decision_uses_the_stable_problem_contract(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, llm_enabled=False)) as client:
        response = client.get("/api/v1/advice/NOT_A_DECISION")
    assert response.status_code == 404
    assert response.json()["code"] == "NOT_FOUND"
    assert response.json()["detail"] == "No guidance exists for that decision."


def test_a_successful_generation_carries_its_disclosure_and_citations(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, llm_enabled=True, handler=_respond(_GOOD_PAYLOAD))) as client:
        response = client.get("/api/v1/advice/WHITE_SPOT_MARKER_DETECTED")
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == "WHITE_SPOT_MARKER_DETECTED"
    assert body["summary"]
    assert body["immediate_actions"]
    assert body["prevention_actions"]
    assert body["review_status"] == "AI_GENERATED_NOT_REVIEWED"
    assert body["review_note"]
    assert body["provider"] == "ollama"
    assert body["model_id"] == "qwen2.5:7b-instruct-q4_0"
    assert body["sources"]
    assert body["based_on_guidance_id"]
    assert body["limitations"]


def test_an_unreachable_ollama_server_answers_503_with_retry_after(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with TestClient(_app(tmp_path, llm_enabled=True, handler=handler)) as client:
        response = client.get("/api/v1/advice/WHITE_SPOT_MARKER_DETECTED")
    assert response.status_code == 503
    assert response.json()["code"] == "ADVICE_UNAVAILABLE"
    assert response.headers["Retry-After"] == "1"


def test_a_response_that_fails_the_safety_scan_answers_503_not_the_raw_text(
    tmp_path: Path,
) -> None:
    unsafe = dict(_GOOD_PAYLOAD, summary="The shrimp is healthy and safe to sell.")
    with TestClient(_app(tmp_path, llm_enabled=True, handler=_respond(unsafe))) as client:
        response = client.get("/api/v1/advice/WHITE_SPOT_MARKER_DETECTED")
    assert response.status_code == 503
    assert response.json()["code"] == "ADVICE_UNAVAILABLE"
    assert "healthy" not in response.text


def test_every_decision_can_be_asked_for_advice(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path, llm_enabled=True, handler=_respond(_GOOD_PAYLOAD))) as client:
        for decision in Decision:
            response = client.get(f"/api/v1/advice/{decision.value}")
            assert response.status_code == 200, decision
