"""Fail-closed validation for the local, cited guidance corpus."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from shrimp_screening.guidance.store import GuidanceError, parse_corpus
from shrimp_screening.paths import guidance_dir


def _document() -> dict[str, Any]:
    return json.loads((guidance_dir() / "guidance_v1.json").read_text(encoding="utf-8"))


def _unknown_source(document: dict[str, Any]) -> None:
    document["items"]["UNABLE_TO_ASSESS"]["source_ids"] = ["not-a-source"]


def _banned_claim(document: dict[str, Any]) -> None:
    document["items"]["UNABLE_TO_ASSESS"]["body"] = "This confirms WSSV infection."


def _missing_decision(document: dict[str, Any]) -> None:
    document["items"].pop("UNABLE_TO_ASSESS")


def _duplicate_guidance_id(document: dict[str, Any]) -> None:
    first = document["items"]["UNABLE_TO_ASSESS"]["id"]
    document["items"]["NO_TARGET_MARKER_DETECTED"]["id"] = first


def _duplicate_source_id(document: dict[str, Any]) -> None:
    document["sources"].append(copy.deepcopy(document["sources"][0]))


def _unknown_decision(document: dict[str, Any]) -> None:
    document["items"]["DISEASE_FREE"] = copy.deepcopy(document["items"]["UNABLE_TO_ASSESS"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_unknown_source, "unknown source"),
        (_banned_claim, "prohibited claim"),
        (_missing_decision, "no guidance exists"),
        (_duplicate_guidance_id, "duplicate guidance ids"),
        (_duplicate_source_id, "duplicate source ids"),
        (_unknown_decision, "not a Decision member"),
    ],
)
def test_corpus_rejects_unsafe_or_incomplete_content(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    document = _document()
    mutate(document)
    with pytest.raises(GuidanceError, match=message):
        parse_corpus(document)


@pytest.mark.parametrize("status", ["", "EXPERT_REVIEWED", True])
def test_corpus_rejects_unearned_review_status(status: object) -> None:
    document = _document()
    document["review_status"] = status
    with pytest.raises(GuidanceError, match="review_status"):
        parse_corpus(document)


def test_corpus_requires_a_human_readable_review_note() -> None:
    document = _document()
    document["review_note"] = ""
    with pytest.raises(GuidanceError, match="review_note"):
        parse_corpus(document)
