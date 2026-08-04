"""Deterministic retrieval over a small, cited, versioned local corpus.

Retrieval is a dictionary lookup keyed on the decision. There is no generative
model anywhere in this path, and that is a safety decision rather than a
shortfall: a lookup cannot invent a dose, a pathogen or a reassurance, and the
exact words a farmer sees are reviewable in Git.

The corpus is validated at load time, not at request time. Every failure below is
a startup failure:

* a decision with no guidance item -- a user would reach a screen with nothing on it;
* a ``source_id`` that resolves to nothing -- an uncheckable citation;
* any prohibited claim (see :mod:`shrimp_screening.guidance.lexicon`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from shrimp_screening.contracts.enums import Decision
from shrimp_screening.guidance.lexicon import assert_clean
from shrimp_screening.paths import data_dir

#: The only review status this corpus is entitled to claim. Widening this set is a
#: deliberate act that should accompany an actual expert review.
ALLOWED_REVIEW_STATUSES = frozenset({"LITERATURE_REVIEWED_NOT_EXPERT_REVIEWED"})


class GuidanceError(RuntimeError):
    """The guidance corpus is missing, malformed, uncited or makes a banned claim."""


@dataclass(frozen=True, slots=True)
class Source:
    """One citable reference."""

    source_id: str
    title: str
    publisher: str
    url: str
    accessed_on: str


@dataclass(frozen=True, slots=True)
class GuidanceItem:
    """The guidance shown for one decision."""

    guidance_id: str
    headline: str
    body: str
    source_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class GuidanceCorpus:
    """The validated corpus, indexed by decision."""

    review_status: str
    review_note: str
    sources: dict[str, Source]
    items: dict[Decision, GuidanceItem]

    def for_decision(self, decision: Decision) -> GuidanceItem:
        return self.items[decision]

    def citations_for(self, decision: Decision) -> tuple[Source, ...]:
        return tuple(self.sources[sid] for sid in self.items[decision].source_ids)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GuidanceError(message)


def _parse_source(entry: Any) -> Source:
    _require(isinstance(entry, dict), "each source must be an object")
    try:
        return Source(
            source_id=str(entry["id"]),
            title=str(entry["title"]),
            publisher=str(entry["publisher"]),
            url=str(entry["url"]),
            accessed_on=str(entry["accessed_on"]),
        )
    except KeyError as exc:
        raise GuidanceError(f"source is missing required field {exc.args[0]!r}") from exc


def parse_corpus(document: Any) -> GuidanceCorpus:
    """Validate a parsed guidance document and return the corpus."""
    _require(isinstance(document, dict), "the guidance corpus must be a JSON object")
    review_status = str(document.get("review_status", ""))
    _require(
        review_status in ALLOWED_REVIEW_STATUSES,
        f"review_status {review_status!r} is not one of {sorted(ALLOWED_REVIEW_STATUSES)}",
    )
    review_note = str(document.get("review_note", ""))
    _require(bool(review_note), "the corpus must carry a human-readable review_note")

    raw_sources = document.get("sources")
    _require(
        isinstance(raw_sources, list) and bool(raw_sources), "sources must be a non-empty array"
    )
    assert isinstance(raw_sources, list)
    sources = {source.source_id: source for source in (_parse_source(e) for e in raw_sources)}
    _require(len(sources) == len(raw_sources), "duplicate source ids")

    raw_items = document.get("items")
    _require(isinstance(raw_items, dict), "items must be an object keyed by decision")
    assert isinstance(raw_items, dict)

    items: dict[Decision, GuidanceItem] = {}
    for key, value in raw_items.items():
        try:
            decision = Decision(key)
        except ValueError as exc:
            raise GuidanceError(f"{key!r} is not a Decision member") from exc
        _require(isinstance(value, dict), f"guidance for {key} must be an object")
        try:
            source_ids = tuple(str(sid) for sid in value["source_ids"])
            item = GuidanceItem(
                guidance_id=str(value["id"]),
                headline=str(value["headline"]),
                body=str(value["body"]),
                source_ids=source_ids,
            )
        except KeyError as exc:
            raise GuidanceError(
                f"guidance for {key} is missing required field {exc.args[0]!r}"
            ) from exc
        _require(bool(item.headline) and bool(item.body), f"guidance for {key} is empty")
        _require(bool(item.source_ids), f"guidance for {key} cites no source")
        for source_id in item.source_ids:
            _require(
                source_id in sources,
                f"guidance for {key} cites unknown source {source_id!r}",
            )
        try:
            assert_clean(f"{item.headline}. {item.body}", where=f"guidance item {item.guidance_id}")
        except ValueError as exc:
            raise GuidanceError(str(exc)) from exc
        items[decision] = item

    missing = sorted(member.value for member in Decision if member not in items)
    _require(not missing, f"no guidance exists for {missing}")
    identifiers = [item.guidance_id for item in items.values()]
    _require(len(set(identifiers)) == len(identifiers), "duplicate guidance ids")

    return GuidanceCorpus(
        review_status=review_status,
        review_note=review_note,
        sources=sources,
        items=items,
    )


@lru_cache(maxsize=1)
def load_guidance(path: Path | None = None) -> GuidanceCorpus:
    """Load and validate ``guidance/guidance_v1.json``."""
    source = path if path is not None else data_dir() / "guidance_v1.json"
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise GuidanceError(f"the guidance corpus could not be read: {source}") from exc
    except json.JSONDecodeError as exc:
        raise GuidanceError(f"the guidance corpus is not valid JSON: {source}") from exc
    return parse_corpus(document)
