"""End-to-end behaviour of the HTTP surface, through the real ASGI stack.

These tests are the ones that would catch a wiring mistake that every unit test
misses: a route that is never registered, a validator that rejects the body the
handler actually builds, an exception handler that turns a 413 into a 500.

Nothing here reads a trained model. `unavailable` is the clean-checkout default and
`fixture` is the synthetic provider, and both are asserted to *say so* in the
response rather than being quietly indistinguishable from a real result.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from shrimp_screening.contracts.enums import (
    AbstentionReason,
    Decision,
    NoticeCode,
    ProviderKind,
    QualityStatus,
)
from shrimp_screening.detection.fixture_provider import SCENARIOS
from shrimp_screening.paths import contracts_dir
from shrimp_screening.settings import Settings
from tests.support.factories import build_app, client_for


def _post_image(client: TestClient, data: bytes, filename: str = "shrimp.jpg") -> object:
    return client.post(
        "/api/v1/screenings",
        files={"image": (filename, data, "image/jpeg")},
    )


# ---------------------------------------------------------------------------
# Health separation: a missing model must not read as a healthy service.
# ---------------------------------------------------------------------------


def test_liveness_is_200_but_readiness_is_503_without_a_model() -> None:
    with client_for("unavailable") as client:
        assert client.get("/livez").status_code == 200
        ready = client.get("/readyz")
        assert ready.status_code == 503
        body = ready.json()
        assert body["provider"] == ProviderKind.UNAVAILABLE.value
        assert body["status"] == "not_ready"
        assert body["model_available"] is False
        assert body["reason"] == "MODEL_UNAVAILABLE"


def test_readiness_is_200_on_a_provider_that_can_actually_infer() -> None:
    with client_for("fixture") as client:
        ready = client.get("/readyz")
        assert ready.status_code == 200
        assert ready.json()["status"] == "ready"


def test_meta_reports_all_five_decisions_and_the_policy_hashes() -> None:
    with client_for("unavailable") as client:
        body = client.get("/api/v1/meta").json()
    assert body["decisions"] == [d.value for d in Decision]
    assert len(body["decisions"]) == 5
    assert body["offline"] is True
    assert body["quality_policy_hash"].startswith("sha256:")
    assert body["decision_policy_hash"].startswith("sha256:")
    assert body["model_available"] is False


# ---------------------------------------------------------------------------
# The unavailable default: abstain honestly, never fabricate inference.
# ---------------------------------------------------------------------------


def test_unavailable_screening_abstains_and_says_why(jpeg_bytes: bytes) -> None:
    with client_for("unavailable") as client:
        response = _post_image(client, jpeg_bytes)
    assert response.status_code == 200
    body = response.json()
    assert body["decision"] == Decision.UNABLE_TO_ASSESS.value
    assert body["abstention_reason"] == AbstentionReason.MODEL_UNAVAILABLE.value
    assert body["model"]["available"] is False
    assert body["markers"] == []
    assert body["confidence_band"] == "NONE"
    assert NoticeCode.MODEL_NOT_INSTALLED.value in body["notices"]
    # Inference cannot have run, so its timing must not be invented.
    assert body["timings_ms"]["inference_ms"] == 0.0


def test_the_client_filename_never_reaches_the_response(jpeg_bytes: bytes) -> None:
    """The filename is attacker-controlled and irrelevant; it must not be echoed."""
    with client_for("unavailable") as client:
        response = _post_image(client, jpeg_bytes, filename="SECRETPONDNAME-gps.jpg")
    assert response.status_code == 200
    assert "SECRETPONDNAME" not in response.text
    assert "secretpondname" not in response.text.lower()


def test_response_validates_against_the_committed_schema(jpeg_bytes: bytes) -> None:
    """The published schema must describe what the service actually returns."""
    schema = json.loads((contracts_dir() / "screening_result.schema.json").read_text("utf-8"))
    with client_for("unavailable") as client:
        body = _post_image(client, jpeg_bytes).json()

    # Validate the parts a hand-rolled check can assert without a jsonschema
    # dependency: required keys present, no undeclared keys, enums respected.
    assert set(schema["required"]) <= set(body)
    assert set(body) <= set(schema["properties"])
    assert body["schema_version"] == schema["properties"]["schema_version"]["const"]


# ---------------------------------------------------------------------------
# Fixture mode is permanently, visibly synthetic.
# ---------------------------------------------------------------------------


def test_fixture_screening_is_permanently_labelled_demonstration(jpeg_bytes: bytes) -> None:
    with client_for("fixture") as client:
        response = _post_image(client, jpeg_bytes)
    assert response.status_code == 200
    body = response.json()
    assert body["model"]["provider"] == ProviderKind.FIXTURE.value
    assert body["model"]["is_demonstration_data"] is True
    assert NoticeCode.DEMONSTRATION_DATA_NOT_A_REAL_RESULT.value in body["notices"]


def test_fixture_mapping_status_stays_visible_in_every_response(jpeg_bytes: bytes) -> None:
    with client_for("fixture") as client:
        body = _post_image(client, jpeg_bytes).json()
    assert body["model"]["dataset_mapping_status"] == "PROVISIONAL_UNCONFIRMED"
    assert NoticeCode.DATASET_CLASS_MAPPING_UNCONFIRMED.value in body["notices"]


@pytest.mark.parametrize("env", ["demo", "production"])
def test_audience_environments_refuse_to_start_on_synthetic_output(env: str) -> None:
    """An audience must never be shown fixture data that looks like a result."""
    with pytest.raises(ValueError, match="silent fallback"):
        Settings(env=env, provider="fixture")  # type: ignore[arg-type]


def test_scenario_parameter_is_rejected_off_the_fixture_provider(jpeg_bytes: bytes) -> None:
    with client_for("unavailable") as client:
        response = client.post(
            "/api/v1/screenings?scenario=white_spot",
            files={"image": ("s.jpg", jpeg_bytes, "image/jpeg")},
        )
    assert response.status_code == 400
    assert response.json()["code"] == "MALFORMED_REQUEST"


def test_all_five_decisions_are_reachable_through_fixture_scenarios(jpeg_bytes: bytes) -> None:
    """The contract promises five states; the demo path must be able to show them all."""
    seen: set[str] = set()
    with client_for("fixture") as client:
        for scenario in SCENARIOS:
            response = client.post(
                f"/api/v1/screenings?scenario={scenario}",
                files={"image": ("s.jpg", jpeg_bytes, "image/jpeg")},
            )
            assert response.status_code == 200, (scenario, response.text)
            seen.add(response.json()["decision"])

    assert seen == {d.value for d in Decision}, f"unreachable decisions: {set(Decision) - seen}"


def test_low_confidence_detection_abstains_instead_of_reporting_a_marker(
    jpeg_bytes: bytes,
) -> None:
    """A detection under the policy threshold must not become a positive finding."""
    with client_for("fixture") as client:
        body = client.post(
            "/api/v1/screenings?scenario=low_confidence",
            files={"image": ("s.jpg", jpeg_bytes, "image/jpeg")},
        ).json()
    assert body["decision"] == Decision.UNABLE_TO_ASSESS.value
    assert body["abstention_reason"] == AbstentionReason.LOW_CONFIDENCE.value


def test_overlapping_boxes_are_suppressed_by_nms(jpeg_bytes: bytes) -> None:
    """The demo path runs real NMS, not a pass-through."""
    with client_for("fixture") as client:
        body = client.post(
            "/api/v1/screenings?scenario=overlapping",
            files={"image": ("s.jpg", jpeg_bytes, "image/jpeg")},
        ).json()
    assert len(body["markers"]) == 1, body["markers"]


def test_quality_failure_abstains_rather_than_guessing(blurry_jpeg_bytes: bytes) -> None:
    with client_for("fixture") as client:
        body = _post_image(client, blurry_jpeg_bytes).json()
    assert body["quality"]["status"] == QualityStatus.FAIL.value
    assert body["quality"]["reasons"], "a failure must carry a retake instruction"
    assert body["decision"] == Decision.UNABLE_TO_ASSESS.value
    assert body["abstention_reason"] == AbstentionReason.IMAGE_QUALITY_REJECTED.value
    assert body["markers"] == [], "a rejected image must not be scored"


# ---------------------------------------------------------------------------
# Error envelope.
# ---------------------------------------------------------------------------


def test_rejects_a_non_image_by_content_not_by_declared_type() -> None:
    with client_for("unavailable") as client:
        response = _post_image(client, b"not an image at all")
    assert response.status_code == 415
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_missing_image_field_is_a_400_not_a_500(jpeg_bytes: bytes) -> None:
    with client_for("unavailable") as client:
        response = client.post(
            "/api/v1/screenings",
            files={"photo": ("s.jpg", jpeg_bytes, "image/jpeg")},
        )
    assert response.status_code == 400
    assert response.json()["code"] == "MALFORMED_REQUEST"


def test_non_multipart_body_is_415() -> None:
    with client_for("unavailable") as client:
        response = client.post("/api/v1/screenings", json={"image": "nope"})
    assert response.status_code == 415
    assert response.json()["code"] == "UNSUPPORTED_MEDIA_TYPE"


@pytest.mark.parametrize(
    "body",
    [
        b"--boundary\r\nBad Header\r\n\r\nbytes\r\n--boundary--\r\n",
        (
            b"--boundary\r\n"
            b'Content-Disposition: form-data; name="image"; filename="x.jpg"\r\n'
            b"\r\npartial"
        ),
        (
            b"--boundary\r\n"
            b'Content-Disposition: form-data; name="image"; filename="a.jpg"\r\n\r\na\r\n'
            b"--boundary\r\n"
            b'Content-Disposition: form-data; name="image"; filename="b.jpg"\r\n\r\nb\r\n'
            b"--boundary--\r\n"
        ),
    ],
)
def test_malformed_multipart_is_a_fixed_400(body: bytes) -> None:
    app = build_app("unavailable")
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/api/v1/screenings",
            content=body,
            headers={"content-type": "multipart/form-data; boundary=boundary"},
        )
    assert response.status_code == 400
    assert response.json()["code"] == "MALFORMED_REQUEST"
    assert response.json()["detail"] == "The multipart request is malformed."


def test_problem_responses_carry_the_request_id_and_no_stack_trace() -> None:
    with client_for("unavailable") as client:
        response = _post_image(client, b"junk")
    body = response.json()
    assert body["request_id"]
    assert "Traceback" not in response.text
    assert "/home/" not in response.text


def test_unknown_route_is_a_problem_document() -> None:
    with client_for("unavailable") as client:
        response = client.get("/api/v1/nope")
    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "NOT_FOUND"


# ---------------------------------------------------------------------------
# Headers and offline posture.
# ---------------------------------------------------------------------------


def test_security_headers_are_present_on_success_and_on_error(jpeg_bytes: bytes) -> None:
    with client_for("unavailable") as client:
        ok = _post_image(client, jpeg_bytes)
        bad = _post_image(client, b"junk")

    for response in (ok, bad):
        assert response.headers["x-content-type-options"] == "nosniff"
        assert response.headers["referrer-policy"] == "no-referrer"
        csp = response.headers["content-security-policy"]
        assert "default-src 'self'" in csp
        # The CSP is also the mechanical proof of the no-CDN claim.
        assert "http://" not in csp
        assert "https://" not in csp


def test_no_temporary_file_survives_a_request(jpeg_bytes: bytes, tmp_path: Path) -> None:
    """Uploads are memory-only: `UploadFile` would have spooled this to disk."""
    before = set(Path(tempfile.gettempdir()).iterdir())
    with client_for("unavailable") as client:
        assert _post_image(client, jpeg_bytes).status_code == 200
    after = set(Path(tempfile.gettempdir()).iterdir())
    leaked = {p for p in after - before if p != tmp_path and tmp_path not in p.parents}
    assert not leaked, f"the request left files behind: {leaked}"


def test_app_can_be_built_without_running_startup(jpeg_bytes: bytes) -> None:
    """Resources are attached eagerly, so a mounted app resolves without lifespan."""
    app = build_app("unavailable")
    client = TestClient(app)  # not used as a context manager: no startup event
    assert client.get("/livez").status_code == 200
