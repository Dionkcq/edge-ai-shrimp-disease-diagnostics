"""Runtime configuration.

Two rules drive the shape of this module:

1. **The safe value is the default.** A clean checkout has no weights, so the
   default provider is ``unavailable`` and the service says so instead of
   pretending to screen.
2. **Demonstration data can never be selected by accident.** ``SHRIMP_ENV=demo``
   or ``production`` refuses to start on anything except a real ONNX provider.
   Fixture output is synthetic; in front of an audience it would be indis-
   tinguishable from a result unless the process simply refuses to run.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["unavailable", "fixture", "onnx"]
EnvironmentName = Literal["dev", "test", "demo", "production"]

#: Hard ceiling on a request body. Chosen to hold a 2048x2048 phone JPEG (1-3 MB)
#: with generous headroom while staying far below anything that would matter for
#: memory on a two-core laptop.
DEFAULT_MAX_UPLOAD_BYTES = 12 * 1024 * 1024

#: Environments in which synthetic output must never be reachable.
AUDIENCE_ENVIRONMENTS: frozenset[str] = frozenset({"demo", "production"})


class Settings(BaseSettings):
    """Environment-driven settings. Every field is prefixed ``SHRIMP_``."""

    model_config = SettingsConfigDict(
        env_prefix="SHRIMP_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        protected_namespaces=(),
    )

    env: EnvironmentName = "dev"
    provider: ProviderName = "unavailable"

    #: Path to an ONNX artifact. Only consulted by the onnx provider, which still
    #: refuses to load it unless its sha256 is registered in models/registry.json.
    onnx_model_path: str | None = None

    max_upload_bytes: int = Field(default=DEFAULT_MAX_UPLOAD_BYTES, ge=1024, le=64 * 1024 * 1024)

    #: One inference at a time. Two ORT sessions on two physical cores is slower
    #: than one, and a queue that is allowed to grow turns a slow request into a
    #: hung browser tab.
    max_concurrent_inferences: int = Field(default=1, ge=1, le=8)
    #: Maximum time a request may wait to enter the inference slot. This does not
    #: bound detector execution: Python worker threads cannot be safely terminated.
    queue_wait_timeout_seconds: float = Field(default=8.0, gt=0.0, le=120.0)

    #: Advertised in the Retry-After header of a 503 SERVICE_BUSY response.
    retry_after_seconds: int = Field(default=2, ge=1, le=120)

    #: Off by default: a clean checkout has no local Ollama server either, and this
    #: feature is additive advice, never the screening decision itself.
    llm_enabled: bool = False
    #: A local Ollama server. Never a remote host by default -- this project ships
    #: no configuration that would send a photograph, a decision or a pond record
    #: off the machine it runs on.
    llm_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:7b-instruct-q4_0"
    #: Generous: a quantized 7B model on modest hardware can take tens of seconds.
    llm_timeout_seconds: float = Field(default=45.0, gt=0.0, le=300.0)

    @model_validator(mode="after")
    def _audience_environments_require_a_real_model(self) -> Self:
        if self.env in AUDIENCE_ENVIRONMENTS and self.provider != "onnx":
            raise ValueError(
                f"SHRIMP_ENV={self.env} refuses to start with SHRIMP_PROVIDER="
                f"{self.provider!r}. Demonstration and production builds must resolve a real "
                "ONNX provider; there is no silent fallback to synthetic output."
            )
        return self


def load_settings() -> Settings:
    """Read settings from the environment. Raises on an unsafe combination."""
    return Settings()
