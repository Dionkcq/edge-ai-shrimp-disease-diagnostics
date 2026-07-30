"""The prohibited-claim scanner.

Half of these tests exist to prove the scanner does *not* fire. A checker that
rejects the project's own disclaimers would push authors toward vaguer language in
exactly the sentences where precision matters most, which is a worse outcome than
having no checker at all.
"""

from __future__ import annotations

import pytest

from shrimp_screening.guidance.lexicon import assert_clean, scan


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("The shrimp is healthy.", "HEALTH_ASSERTION"),
        ("These animals are disease-free.", "HEALTH_ASSERTION"),
        ("The stock is safe to sell.", "HEALTH_ASSERTION"),
        ("This app will diagnose white spot syndrome.", "DIAGNOSIS"),
        ("Use it as a diagnostic test.", "DIAGNOSIS"),
        ("The photograph confirms WSSV.", "PATHOGEN_CONFIRMATION"),
        ("A marked spot proves an infection.", "PATHOGEN_CONFIRMATION"),
        ("Treat the pond immediately.", "CURE_OR_TREATMENT"),
        ("This will cure the outbreak.", "CURE_OR_TREATMENT"),
        ("Apply oxytetracycline to the pond.", "DRUG_NAMED"),
        ("Dose at 5 mg/L.", "DOSE_QUANTITY"),
        ("Maintain 2.5 ppm.", "DOSE_QUANTITY"),
        ("The system detects EMS.", "EMS_AHPND_CLAIM"),
        ("This tool may diagnose WSSV.", "DIAGNOSIS"),
        ("This tool only diagnoses WSSV.", "DIAGNOSIS"),
        ("Ask a professional because this image confirms WSSV.", "PATHOGEN_CONFIRMATION"),
        ("This does not assess lighting and confirms WSSV.", "PATHOGEN_CONFIRMATION"),
        ("It might detect AHPND.", "EMS_AHPND_CLAIM"),
        ("There is no doubt the shrimp is healthy.", "HEALTH_ASSERTION"),
    ],
)
def test_a_claim_the_system_cannot_support_is_flagged(text: str, code: str) -> None:
    assert code in {violation.code for violation in scan(text)}


@pytest.mark.parametrize(
    "text",
    [
        "This result does not mean the shrimp is healthy or disease-free.",
        "A photograph cannot confirm an infection.",
        "This is not a diagnosis and must not be used as one.",
        "The image screen never confirms white spot syndrome virus.",
        "No result here proves a disease is present.",
        "This system does not detect EMS or AHPND.",
        "Do not treat the pond on the basis of this screen alone.",
        "Ask a qualified aquatic-animal health professional whether sampling is warranted.",
        "Reflections, shell texture and debris can resemble spots.",
        "Treat this as a reason to document and escalate, not as confirmation of an infection.",
        "Continue routine observation and use professional assessment when mortality is "
        "concerning.",
    ],
)
def test_a_legitimate_disclaimer_is_not_flagged(text: str) -> None:
    assert scan(text) == [], f"the scanner rejected a required disclaimer: {text!r}"


def test_a_dose_is_prohibited_even_inside_a_warning() -> None:
    """Naming a compound and a number is how a reader ends up with a dose, whatever
    the surrounding words say. The correct phrasing contains neither."""
    assert scan("Do not apply formalin at 25 ppm without advice.") != []


def test_the_scanner_reports_where_the_claim_was_made() -> None:
    violations = scan("Everything is fine. The shrimp is healthy. Ask a professional.")
    assert len(violations) == 1
    assert "The shrimp is healthy." in violations[0].sentence
    assert violations[0].rationale


def test_assert_clean_names_the_location_and_every_violation() -> None:
    with pytest.raises(ValueError, match="guidance item xyz") as caught:
        assert_clean("The shrimp is healthy. Dose at 5 mg/L.", where="guidance item xyz")
    message = str(caught.value)
    assert "HEALTH_ASSERTION" in message
    assert "DOSE_QUANTITY" in message


def test_clean_text_passes_silently() -> None:
    assert_clean("Retake the photograph under even light.", where="test")
