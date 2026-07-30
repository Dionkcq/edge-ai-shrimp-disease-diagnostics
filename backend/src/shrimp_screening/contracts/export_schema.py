"""Generate the committed JSON Schema artifacts from the Pydantic models.

Run via ``make schema`` or ``uv run shrimp-export-schema``. A contract test
regenerates and byte-compares, so forgetting to run this is a CI failure rather than
a silent divergence between the backend and any generated client.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from shrimp_screening.contracts.guidance import GuidanceDocument
from shrimp_screening.contracts.problem import ProblemDetail
from shrimp_screening.contracts.screening import SCHEMA_VERSION, ScreeningResult
from shrimp_screening.paths import repository_root

JSON_SCHEMA_DIALECT = "https://json-schema.org/draft/2020-12/schema"


@dataclass(frozen=True, slots=True)
class SchemaArtifact:
    """One generated, committed schema file."""

    relative_path: str
    model: type[BaseModel]
    schema_id: str
    description: str


SCHEMA_ARTIFACTS: tuple[SchemaArtifact, ...] = (
    SchemaArtifact(
        relative_path="contracts/screening_result.schema.json",
        model=ScreeningResult,
        schema_id="https://shrimp-screening.invalid/schemas/screening_result.schema.json",
        description=(
            "Response body of POST /api/v1/screenings. Generated from "
            "backend/src/shrimp_screening/contracts/screening.py; do not hand-edit."
        ),
    ),
    SchemaArtifact(
        relative_path="contracts/problem_detail.schema.json",
        model=ProblemDetail,
        schema_id="https://shrimp-screening.invalid/schemas/problem_detail.schema.json",
        description=(
            "RFC 9457 problem document returned for every 4xx/5xx API response. "
            "Generated from backend/src/shrimp_screening/contracts/problem.py; "
            "do not hand-edit."
        ),
    ),
    SchemaArtifact(
        relative_path="contracts/guidance_document.schema.json",
        model=GuidanceDocument,
        schema_id="https://shrimp-screening.invalid/schemas/guidance_document.schema.json",
        description=(
            "Response body of GET /api/v1/guidance/{decision}. Generated from "
            "backend/src/shrimp_screening/contracts/guidance.py; do not hand-edit."
        ),
    ),
)


def build_schema(artifact: SchemaArtifact) -> dict[str, object]:
    """Return the JSON Schema document for one artifact."""
    schema = artifact.model.model_json_schema(mode="serialization")
    schema["$schema"] = JSON_SCHEMA_DIALECT
    schema["$id"] = artifact.schema_id
    schema["description"] = artifact.description
    schema["x-contract-version"] = SCHEMA_VERSION
    return schema


def render_schema(artifact: SchemaArtifact) -> str:
    """Return the exact file content for one artifact, including the trailing newline."""
    return json.dumps(build_schema(artifact), indent=2, sort_keys=True) + "\n"


def write_all(root: Path | None = None) -> list[Path]:
    """Write every artifact under ``root`` (default: the repository root)."""
    base = root if root is not None else repository_root()
    written: list[Path] = []
    for artifact in SCHEMA_ARTIFACTS:
        target = base / artifact.relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_schema(artifact), encoding="utf-8")
        written.append(target)
    return written


def main() -> int:
    for path in write_all():
        print(f"wrote {path}")  # this is a CLI; stdout is the interface
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
