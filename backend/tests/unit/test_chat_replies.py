"""Image-result replies fail closed to the cited local guidance."""

from __future__ import annotations

from typing import Any

import pytest

from shrimp_screening.ai.replies import guard_chat_reply


def _screened_result() -> dict[str, Any]:
    return {
        "status": "screened",
        "decision": "WHITE_SPOT_MARKER_DETECTED",
        "guidance": {
            "headline": "A white-spot-like region was marked",
            "body": "Separate questionable stock and record pond conditions for review.",
            "sources": [{"title": "WOAH Aquatic Manual"}],
        },
    }


def test_image_result_always_uses_cited_guidance() -> None:
    reply = (
        "The screen marked a white-spot-like region. This is not a diagnosis; "
        "ask an aquatic-animal health professional to review the pond."
    )

    guarded = guard_chat_reply(reply, _screened_result())

    assert guarded != reply
    assert "Separate questionable stock" in guarded
    assert "WOAH Aquatic Manual" in guarded


def test_guard_replaces_prohibited_medication_and_dose() -> None:
    reply = "Treat the shrimp with oxytetracycline at 5 mg/L."

    guarded = guard_chat_reply(reply, _screened_result())

    assert "oxytetracycline" not in guarded.lower()
    assert "5 mg/l" not in guarded.lower()
    assert "Separate questionable stock" in guarded
    assert "aquatic-animal health professional" in guarded
    assert "WOAH Aquatic Manual" in guarded


def test_fallback_does_not_repeat_an_existing_aquatic_referral() -> None:
    result = _screened_result()
    result["guidance"]["body"] = (
        "Record pond conditions and ask a qualified aquatic-animal health professional "
        "whether sampling is warranted."
    )

    guarded = guard_chat_reply("Ask a qualified healthcare professional.", result)

    assert guarded.lower().count("aquatic-animal health professional") == 1


def test_treatment_intent_always_uses_cited_guidance() -> None:
    ungrounded_but_plausible = "Increase the pond salinity and change the feed immediately."

    guarded = guard_chat_reply(
        ungrounded_but_plausible,
        _screened_result(),
        user_message="How do I treat this?",
    )

    assert "Increase the pond salinity" not in guarded
    assert "Separate questionable stock" in guarded
    assert "WOAH Aquatic Manual" in guarded


def test_fallback_surfaces_guidance_review_note() -> None:
    result = _screened_result()
    result["guidance"]["review_note"] = (
        "Assembled from cited sources by project authors; not expert-reviewed."
    )

    guarded = guard_chat_reply("Ask a doctor.", result)

    assert "not expert-reviewed" in guarded


@pytest.mark.parametrize(
    "reply",
    (
        "Seek medical attention at a clinic.",
        "Consult a clinician or healthcare provider.",
        "Go to the hospital.",
        "Ask doctors or physicians for advice.",
        "Contact emergency services.",
    ),
)
def test_human_health_variants_fail_closed_without_a_tool_result(reply: str) -> None:
    guarded = guard_chat_reply(reply, None)

    assert guarded != reply
    assert "shrimp image screening" in guarded


@pytest.mark.parametrize(
    "message",
    (
        "What can I do?",
        "How should I manage this?",
        "What action should I take?",
        "Any recommendations?",
        "How can I help them?",
        "Do they need medications?",
        "What are the next steps?",
        "Should I change the pond water?",
        "Tell me how to fix this.",
        "What would you recommend doing now?",
        "How can I keep them alive?",
        "Hello, can you help me?",
    ),
)
def test_no_image_action_requests_ask_for_a_shrimp_photo(message: str) -> None:
    guarded = guard_chat_reply(
        "Increase salinity and replace some pond water.",
        None,
        user_message=message,
    )

    assert "Increase salinity" not in guarded
    assert "Upload a shrimp photograph" in guarded


def test_needs_image_and_malformed_guidance_fail_closed() -> None:
    needs_image = guard_chat_reply(
        "Anything goes.", {"status": "needs_image"}, user_message="What should I do?"
    )
    malformed = guard_chat_reply("Anything goes.", {"status": "screened", "guidance": "bad"})

    assert "upload a shrimp photograph" in needs_image.lower()
    assert "did not include" in malformed
