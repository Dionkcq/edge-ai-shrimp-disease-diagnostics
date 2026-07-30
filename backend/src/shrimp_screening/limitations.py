"""``docs/LIMITATIONS.md`` is the source of the API's ``limitations[]`` array.

Keeping the prose and the emitted identifiers in one file means the two cannot
drift: there is no second list in Python to forget to update. A contract test
asserts that every identifier a response can emit is defined in the document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from shrimp_screening.contracts.enums import Decision
from shrimp_screening.paths import docs_dir

_HEADING = re.compile(r"^##\s+(lim-[a-z0-9-]+)\s*$", re.MULTILINE)
_APPLIES_TO = re.compile(r"^\*\*Applies to:\*\*\s*(.+?)\s*$", re.MULTILINE)

#: Sentinel in the `Applies to:` line meaning "every decision".
_ALL = "all"


class LimitationsError(RuntimeError):
    """The limitations document is missing or does not parse."""


@dataclass(frozen=True, slots=True)
class Limitation:
    """One declared limitation and the decisions it attaches to."""

    limitation_id: str
    decisions: frozenset[Decision]
    text: str

    def applies_to(self, decision: Decision) -> bool:
        return decision in self.decisions


def _parse_decisions(raw: str, limitation_id: str) -> frozenset[Decision]:
    if raw.strip() == _ALL:
        return frozenset(Decision)
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise LimitationsError(f"{limitation_id} declares no decisions")
    try:
        return frozenset(Decision(name) for name in names)
    except ValueError as exc:
        raise LimitationsError(f"{limitation_id} names an unknown decision: {exc}") from exc


def parse_limitations(document: str) -> tuple[Limitation, ...]:
    """Parse the markdown document into limitation records."""
    matches = list(_HEADING.finditer(document))
    if not matches:
        raise LimitationsError("no '## lim-...' headings were found")
    parsed: list[Limitation] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(document)
        body = document[match.end() : end]
        applies = _APPLIES_TO.search(body)
        if applies is None:
            raise LimitationsError(f"{match.group(1)} has no '**Applies to:**' line")
        text = _APPLIES_TO.sub("", body).strip()
        if not text:
            raise LimitationsError(f"{match.group(1)} has no explanatory text")
        parsed.append(
            Limitation(
                limitation_id=match.group(1),
                decisions=_parse_decisions(applies.group(1), match.group(1)),
                text=text,
            )
        )
    identifiers = [item.limitation_id for item in parsed]
    if len(set(identifiers)) != len(identifiers):
        raise LimitationsError("duplicate limitation identifiers")
    return tuple(parsed)


@lru_cache(maxsize=1)
def load_limitations(path: Path | None = None) -> tuple[Limitation, ...]:
    source = path if path is not None else docs_dir() / "LIMITATIONS.md"
    try:
        document = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise LimitationsError(f"the limitations document could not be read: {source}") from exc
    return parse_limitations(document)


def limitation_ids_for(decision: Decision, path: Path | None = None) -> list[str]:
    """Identifiers that must accompany one decision, in document order."""
    return [item.limitation_id for item in load_limitations(path) if item.applies_to(decision)]
