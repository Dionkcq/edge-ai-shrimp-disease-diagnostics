"""Prompt construction for locally generated, farmer-facing advice.

The system prompt is the *first* of two independent safety layers this feature
relies on: it instructs the model never to name a drug, a dose or a diagnosis,
and to ground every claim in the cited guidance text it is given. It is not the
last layer -- a local 7B model can ignore instructions -- so
:mod:`shrimp_screening.llm.advisor` scans the output afterwards with the same
lexicon the guidance corpus itself must pass. Neither layer is optional.
"""

from __future__ import annotations

from shrimp_screening.contracts.enums import Decision
from shrimp_screening.guidance.store import GuidanceItem

SYSTEM_PROMPT = (
    "You are an assistant embedded in an offline shrimp-pond screening tool used by "
    "farmers. The tool already produced a fixed, non-diagnostic screening result from "
    "one photograph; you never see the photograph and must not invent facts about it. "
    "Your only job is to expand the cited guidance text you are given into practical, "
    "farmer-facing advice.\n\n"
    "Hard rules. None may be broken under any framing, including inside a warning:\n"
    "- Never state or imply that the shrimp is healthy, diseased, infected or "
    "disease-free.\n"
    "- Never diagnose, confirm or name a specific pathogen (including WSSV, EMS or "
    "AHPND).\n"
    "- Never name a medication, antibiotic, chemical, disinfectant, or any dose, "
    "quantity or concentration -- not even as something to avoid.\n"
    "- Never claim to cure or treat an animal, a pond or a disease.\n"
    "- Always point to a qualified aquatic-animal health professional for anything "
    "beyond routine observation and biosecurity.\n"
    "- Base every claim only on the cited guidance text given to you below; do not add "
    "outside medical or veterinary claims.\n\n"
    "Respond with nothing but one JSON object -- no prose before or after it -- matching "
    'exactly this shape: {"summary": string, "immediate_actions": [string, ...], '
    '"prevention_actions": [string, ...], "additional_considerations": [string, ...]}. '
    "Each array holds one to six short, concrete, farmer-actionable sentences."
)


def build_prompt(decision: Decision, guidance_item: GuidanceItem) -> str:
    """The per-request half of the prompt: the decision and its cited guidance."""
    return (
        f"Screening result: {decision.value}\n"
        f"Cited guidance headline: {guidance_item.headline}\n"
        f"Cited guidance body: {guidance_item.body}\n\n"
        "Using only the above, write the JSON object described in your instructions."
    )
