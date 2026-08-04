from pathlib import Path

import httpx2 as httpx
from fastapi.testclient import TestClient
from PIL import Image

from shrimp_screening.llm.client import OllamaClient
from shrimp_screening.main import create_app
from shrimp_screening.settings import Settings


def _settings() -> Settings:
    return Settings(
        env="test",
        provider="unavailable",
        onnx_model_path=None,
        max_upload_bytes=2 * 1024 * 1024,
        max_concurrent_inferences=1,
        queue_wait_timeout_seconds=1.0,
        retry_after_seconds=1,
        llm_enabled=True,
    )


def _image_bytes(tmp_path: Path) -> bytes:
    path = tmp_path / "shrimp.png"
    Image.new("RGB", (64, 64), (120, 120, 120)).save(path)
    return path.read_bytes()


def test_chat_runs_model_tool_model_loop_and_returns_memory(tmp_path: Path) -> None:
    responses = [
        {
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "screen_shrimp_image",
                            "arguments": {"reason": "inspect the uploaded photo"},
                        }
                    }
                ],
            }
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json=responses.pop(0))

    llm_client = OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:7b-instruct-q4_0",
        timeout_seconds=1.0,
        transport=httpx.MockTransport(handler),
    )
    app = create_app(
        _settings(), llm_client=llm_client, frontend_dir=tmp_path / "no-frontend-build"
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            data={"message": "Please inspect this photo"},
            files={"image": ("shrimp.png", _image_bytes(tmp_path), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["reply"].startswith("The image could not be assessed.")
    assert body["tool_result"]["guidance"]["body"] in body["reply"]
    assert "Declared limitations of this screening system" in body["reply"]
    assert body["tool_result"]["guidance"]["review_note"] in body["reply"]
    assert body["tool_calls"] == [{"name": "screen_shrimp_image", "status": "completed"}]
    assert body["tool_result"]["status"] == "abstained"
    assert body["tool_result"]["guidance"]["decision"] == "UNABLE_TO_ASSESS"
    assert body["tool_result"]["guidance"]["body"]
    assert body["tool_result"]["guidance"]["sources"] == [
        {
            "id": "project-limitations",
            "title": "Declared limitations of this screening system",
            "publisher": "OIP Group One",
            "url": "docs/LIMITATIONS.md",
        }
    ]
    assert [message["role"] for message in body["messages"]] == ["user", "tool", "assistant"]
    assert responses == []
