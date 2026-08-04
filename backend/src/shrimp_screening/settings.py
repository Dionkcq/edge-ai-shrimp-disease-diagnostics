"""The single global configuration module.

Everything tunable about a running service lives here: the environment-driven
:class:`Settings` and, above it, the fixed limits that bound the process. They are
kept together so that "what can I turn, and how far" is one file rather than a
hunt through the module that happens to use each value.

Two rules drive the shape of :class:`Settings`:

1. **The safe value is the default.** A clean checkout has no weights, so the
   default provider is ``unavailable`` and the service says so instead of
   pretending to screen.
2. **Demonstration data can never be selected by accident.** ``SHRIMP_ENV=demo``
   or ``production`` refuses to start on anything except a real ONNX provider.
   Fixture output is synthetic; in front of an audience it would be indis-
   tinguishable from a result unless the process simply refuses to run.

What deliberately stays *out* of this file: values that are not configuration at
all but invariants of a format or an algorithm -- the JPEG/PNG magic bytes, the
letterbox pad value of 114, the YOLO stride and anchor layout, the guidance
lexicon's patterns. Those cannot be retuned without changing the code that
depends on them, so they belong next to that code, where a reader will see them.

This module imports nothing from the rest of the package, so it is safe to import
from anywhere in it.
"""

from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["unavailable", "fixture", "onnx"]
EnvironmentName = Literal["dev", "test", "demo", "production"]

# ---------------------------------------------------------------------------
# Service identity, published in the OpenAPI document.
# ---------------------------------------------------------------------------

API_TITLE = "Edge AI Shrimp Visible-Marker Screening"
API_VERSION = "0.1.0"
API_DESCRIPTION = (
    "Offline screening for two visible appearances in a shrimp photograph. This "
    "service reports appearances, never a diagnosis, and abstains rather than "
    "guessing. On a checkout with no trained weights the default provider is "
    "'unavailable': /readyz answers 503 and every screening answers "
    "UNABLE_TO_ASSESS / MODEL_UNAVAILABLE."
)

# ---------------------------------------------------------------------------
# Upload and decode limits.
#
# These bound work done on untrusted input, so they are fixed rather than
# environment-driven: an operator who could raise them from the environment could
# also undo the memory guarantees the intake path exists to provide.
# ---------------------------------------------------------------------------

#: Hard ceiling on a request body. Chosen to hold a 2048x2048 phone JPEG (1-3 MB)
#: with generous headroom while staying far below anything that would matter for
#: memory on a two-core laptop. ``Settings.max_upload_bytes`` defaults to this and
#: ``decode_image`` uses it directly, so a script calling the decoder is bounded
#: identically to a request arriving over HTTP.
DEFAULT_MAX_UPLOAD_BYTES = 12 * 1024 * 1024

#: 40 Mpx is comfortably above a 2048x2048 phone capture and far below the point
#: where a decode would matter for memory on a two-core laptop.
MAX_IMAGE_PIXELS = 40_000_000
MAX_DIMENSION = 8_000

#: A screening request carries exactly one file. Anything else is a client bug or
#: an attempt to make the multipart parser do work proportional to attacker input.
MAX_PARTS = 8

#: Bound on the ``name="..."`` bookkeeping buffered per multipart part.
MAX_HEADER_BYTES = 4096

# ---------------------------------------------------------------------------
# Inference limits.
# ---------------------------------------------------------------------------

#: Two physical cores is the target machine; more ORT threads than that is slower.
INTRA_OP_THREADS = 2

#: Ceiling on boxes entering NMS. A corrupt or adversarial output tensor must not
#: turn a decode into a quadratic sort.
MAX_NMS_CANDIDATES = 30_000

# ---------------------------------------------------------------------------
# Local LLM limits. The feature is additive advice, never the screening decision.
# ---------------------------------------------------------------------------

#: A local Ollama server. Never a remote host by default -- this project ships no
#: configuration that would send a photograph, a decision or a pond record off the
#: machine it runs on. ``Settings.llm_base_url`` defaults to this, and so does a
#: directly constructed ``OllamaClient``.
DEFAULT_LLM_BASE_URL = "http://127.0.0.1:11434"

#: Truncation applied to each advice list before it reaches a reader.
MAX_ADVICE_ITEMS_PER_LIST = 6

#: Turns of conversation retained per session. Bounded so a long chat cannot grow
#: process memory without limit.
MAX_CHAT_MEMORY_MESSAGES = 12

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
    llm_base_url: str = DEFAULT_LLM_BASE_URL
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
