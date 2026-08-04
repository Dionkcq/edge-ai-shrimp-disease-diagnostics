"""The cited guidance endpoint is a safety-bearing part of every result."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from shrimp_screening.contracts.enums import Decision
from shrimp_screening.settings import Settings
from shrimp_server.main import create_app


def _app(tmp_path: Path) -> FastAPI:
    settings = Settings(
        env="test",
        provider="unavailable",
        onnx_model_path=None,
        max_upload_bytes=2 * 1024 * 1024,
        max_concurrent_inferences=1,
        queue_wait_timeout_seconds=1.0,
        retry_after_seconds=1,
    )
    return create_app(settings, frontend_dir=tmp_path / "no-frontend-build")


def test_every_decision_has_cited_non_expert_reviewed_guidance(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        for decision in Decision:
            response = client.get(f"/api/v1/guidance/{decision.value}")
            assert response.status_code == 200
            document = response.json()
            assert document["decision"] == decision.value
            assert document["sources"]
            assert all(
                source["id"] and source["title"] and source["url"] for source in document["sources"]
            )
            assert document["review_status"] == "LITERATURE_REVIEWED_NOT_EXPERT_REVIEWED"
            assert document["review_note"]
            assert document["limitations"]


def test_negative_marker_guidance_retains_the_not_healthy_limitation(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        response = client.get("/api/v1/guidance/NO_TARGET_MARKER_DETECTED")
    assert response.status_code == 200
    assert "lim-negative-is-not-health" in response.json()["limitations"]


def test_unknown_guidance_uses_the_stable_problem_contract(tmp_path: Path) -> None:
    with TestClient(_app(tmp_path)) as client:
        response = client.get("/api/v1/guidance/NOT_A_DECISION")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "NOT_FOUND"
    assert response.json()["detail"] == "No guidance exists for that decision."
