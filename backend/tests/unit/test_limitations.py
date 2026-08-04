"""The limitations table is a safety disclosure, so its invariants are asserted.

These were previously enforced by the markdown parser (duplicate identifiers and
empty bodies raised at startup). The data is literal now, so the same guarantees
have to be tested instead.
"""

from __future__ import annotations

import pytest

from shrimp_screening.contracts.enums import Decision
from shrimp_screening.limitations import LIMITATIONS, limitation_ids_for, load_limitations


def test_identifiers_are_unique() -> None:
    identifiers = [item.limitation_id for item in LIMITATIONS]
    assert len(set(identifiers)) == len(identifiers)


def test_every_identifier_uses_the_declared_prefix() -> None:
    """The `lim-` prefix is part of the published contract, not a naming habit."""
    for item in LIMITATIONS:
        assert item.limitation_id.startswith("lim-"), item.limitation_id


def test_no_limitation_has_empty_text_or_no_decisions() -> None:
    for item in LIMITATIONS:
        assert item.text.strip(), f"{item.limitation_id} has no explanatory text"
        assert item.decisions, f"{item.limitation_id} applies to no decision"


@pytest.mark.parametrize("decision", list(Decision))
def test_every_decision_carries_at_least_one_limitation(decision: Decision) -> None:
    """No response may go out with an empty `limitations[]` array."""
    assert limitation_ids_for(decision)


def test_the_unconditional_disclosures_reach_every_decision() -> None:
    """These are the ones that must never be omitted, whatever the outcome."""
    unconditional = {
        "lim-not-diagnostic",
        "lim-no-lab-confirmation",
        "lim-two-markers-only",
    }
    for decision in Decision:
        assert unconditional <= set(limitation_ids_for(decision)), decision


def test_a_negative_result_is_disclosed_as_not_a_clean_bill_of_health() -> None:
    ids = limitation_ids_for(Decision.NO_TARGET_MARKER_DETECTED)
    assert "lim-negative-is-not-health" in ids


def test_load_limitations_returns_the_table_in_declaration_order() -> None:
    assert load_limitations() == LIMITATIONS
