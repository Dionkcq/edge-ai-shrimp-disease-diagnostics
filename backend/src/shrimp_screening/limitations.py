"""The source of the API's ``limitations[]`` array.

This is not commentary. Every screening, guidance and advice response carries the
identifiers selected here, and they are the product's safety disclosure: the
statement that a negative result is not a clean bill of health, that no laboratory
method was involved, that only two appearances are in scope. Removing an entry
removes a disclosure from a live response, so entries are added and removed
deliberately, not tidied.

These records used to live in a markdown document that this module parsed at
startup. They are literal data now, which means a malformed entry is a type error
at import rather than a runtime failure on a machine where the file was missing.
"""

from __future__ import annotations

from dataclasses import dataclass

from shrimp_screening.contracts.enums import Decision

#: Every decision, for limitations that are unconditional.
_ALL: frozenset[Decision] = frozenset(Decision)


@dataclass(frozen=True, slots=True)
class Limitation:
    """One declared limitation and the decisions it attaches to."""

    limitation_id: str
    decisions: frozenset[Decision]
    text: str

    def applies_to(self, decision: Decision) -> bool:
        return decision in self.decisions


#: Order is response order: unconditional disclosures first, then the ones that
#: qualify a specific decision.
LIMITATIONS: tuple[Limitation, ...] = (
    Limitation(
        limitation_id="lim-not-diagnostic",
        decisions=_ALL,
        text=(
            "This system reports visible appearances in a photograph. It does not "
            "identify a pathogen, does not establish a cause and is not a diagnosis. "
            "Any result that matters should be confirmed by a qualified "
            "aquatic-animal health professional."
        ),
    ),
    Limitation(
        limitation_id="lim-no-lab-confirmation",
        decisions=_ALL,
        text=(
            "No laboratory method is involved. Confirmation of white spot syndrome "
            "virus requires molecular or histological testing that an image cannot "
            "substitute for."
        ),
    ),
    Limitation(
        limitation_id="lim-two-markers-only",
        decisions=_ALL,
        text=(
            "Only two visible appearances are in scope: white-spot-like regions and "
            "dark-gill-like regions. Every other condition, including EMS/AHPND, "
            "yellow head and tail discoloration, is outside the intended screening "
            "scope and will not be reported even when present."
        ),
    ),
    Limitation(
        limitation_id="lim-mapping-provisional",
        decisions=_ALL,
        text=(
            "The class order of the combined-folder annotations in the source dataset "
            "was inferred from image evidence, not confirmed by the dataset authors. "
            "Responses carry `model.dataset_mapping_status` for as long as that "
            "remains true."
        ),
    ),
    Limitation(
        limitation_id="lim-no-ood-gate",
        decisions=_ALL,
        text=(
            "There is no check that the photograph contains a shrimp at all. The "
            "system will score a photograph of a hand, a net or a bucket and report "
            "an appearance-based result for it."
        ),
    ),
    Limitation(
        limitation_id="lim-uncalibrated-thresholds",
        decisions=_ALL,
        text=(
            "Every score threshold and quality threshold in `policy/` is an unfitted "
            "starting point. No threshold in this repository has been calibrated "
            "against ground truth, because no trained model exists."
        ),
    ),
    Limitation(
        limitation_id="lim-negative-is-not-health",
        decisions=frozenset({Decision.NO_TARGET_MARKER_DETECTED}),
        text=(
            'A result of "no target marker detected" does not mean the shrimp is '
            "healthy or disease-free. It means the two appearances in scope were not "
            "found in this one photograph, at this one angle, under this lighting."
        ),
    ),
    Limitation(
        limitation_id="lim-small-target-scale",
        decisions=frozenset(
            {
                Decision.WHITE_SPOT_MARKER_DETECTED,
                Decision.MULTIPLE_TARGET_MARKERS_DETECTED,
            }
        ),
        text=(
            "White-spot targets in the source data have a median size of roughly "
            "eleven pixels at the model's input resolution, which is at the edge of "
            "what a compact detector can resolve. Both missed spots and spurious ones "
            "are expected until this is measured."
        ),
    ),
    Limitation(
        limitation_id="lim-single-photograph",
        decisions=frozenset(
            {
                Decision.GILL_DARKENING_MARKER_DETECTED,
                Decision.WHITE_SPOT_MARKER_DETECTED,
                Decision.MULTIPLE_TARGET_MARKERS_DETECTED,
            }
        ),
        text=(
            "One photograph of one animal is not a pond assessment. Shell texture, "
            "reflections, debris and shadow all resemble the appearances being "
            "screened for."
        ),
    ),
    Limitation(
        limitation_id="lim-no-model-installed",
        decisions=frozenset({Decision.UNABLE_TO_ASSESS}),
        text=(
            "When no screening model is installed, the service still validates and "
            "measures the photograph but performs no inference at all. Nothing about "
            "the animal is being assessed in that state."
        ),
    ),
)


def load_limitations() -> tuple[Limitation, ...]:
    """Return every declared limitation, in response order."""
    return LIMITATIONS


def limitation_ids_for(decision: Decision) -> list[str]:
    """Identifiers that must accompany one decision, in declaration order."""
    return [item.limitation_id for item in LIMITATIONS if item.applies_to(decision)]
