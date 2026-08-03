"""Configuration must make the unsafe combination impossible, not merely unlikely."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shrimp_screening.settings import Settings, load_settings


def test_the_default_is_the_safe_one() -> None:
    settings = Settings()
    assert settings.provider == "unavailable"
    assert settings.env == "dev"
    assert settings.max_upload_bytes == 12 * 1024 * 1024
    assert settings.llm_enabled is False


@pytest.mark.parametrize("environment", ["demo", "production"])
@pytest.mark.parametrize("provider", ["fixture", "unavailable"])
def test_an_audience_environment_refuses_to_start_without_a_real_model(
    environment: str, provider: str
) -> None:
    """Synthetic output in front of an audience is indistinguishable from a result.

    The only defence that survives a stressful demonstration is the process
    refusing to start at all.
    """
    with pytest.raises(ValidationError, match="refuses to start"):
        Settings(env=environment, provider=provider)  # type: ignore[arg-type]


def test_a_development_environment_may_use_the_fixture_provider() -> None:
    assert Settings(env="dev", provider="fixture").provider == "fixture"
    assert Settings(env="test", provider="fixture").provider == "fixture"


def test_settings_are_read_from_the_shrimp_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHRIMP_PROVIDER", "fixture")
    monkeypatch.setenv("SHRIMP_MAX_UPLOAD_BYTES", "2097152")
    monkeypatch.setenv("SHRIMP_QUEUE_WAIT_TIMEOUT_SECONDS", "3.5")
    settings = load_settings()
    assert settings.provider == "fixture"
    assert settings.max_upload_bytes == 2 * 1024 * 1024
    assert settings.queue_wait_timeout_seconds == 3.5


def test_an_absurd_upload_cap_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(max_upload_bytes=1024 * 1024 * 1024)


def test_llm_settings_are_read_from_the_shrimp_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SHRIMP_LLM_ENABLED", "true")
    monkeypatch.setenv("SHRIMP_LLM_BASE_URL", "http://127.0.0.1:12345")
    monkeypatch.setenv("SHRIMP_LLM_MODEL", "qwen2.5:7b-instruct-q4_0")
    monkeypatch.setenv("SHRIMP_LLM_TIMEOUT_SECONDS", "10")
    settings = load_settings()
    assert settings.llm_enabled is True
    assert settings.llm_base_url == "http://127.0.0.1:12345"
    assert settings.llm_model == "qwen2.5:7b-instruct-q4_0"
    assert settings.llm_timeout_seconds == 10.0


def test_an_absurd_llm_timeout_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(llm_timeout_seconds=0.0)
    with pytest.raises(ValidationError):
        Settings(llm_timeout_seconds=1000.0)


def test_settings_are_immutable_once_loaded() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.provider = "fixture"  # type: ignore[misc]
