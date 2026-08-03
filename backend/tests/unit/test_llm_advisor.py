"""Two independent safety layers guard generated advice: a prompt (untested here,
since a prompt is a request, not a guarantee) and the output scan below, which is
the actual gate. These tests exercise the gate against a model that is not
cooperating -- bad JSON, missing fields, and text a real 7B model could plausibly
produce despite being told not to.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import httpx2 as httpx
import pytest

from shrimp_screening.contracts.enums import Decision
from shrimp_screening.guidance.store import GuidanceItem
from shrimp_screening.llm.advisor import AdviceGenerationError, generate_advice
from shrimp_screening.llm.client import OllamaClient

Handler = Callable[[httpx.Request], httpx.Response]

_GUIDANCE_ITEM = GuidanceItem(
    guidance_id="guide-white-spot-v1",
    headline="A white-spot-like region was marked",
    body="Repeat the capture on more than one shrimp and escalate to a professional.",
    source_ids=("woah-wssv-code",),
)


def _client(handler: Handler) -> OllamaClient:
    return OllamaClient(
        base_url="http://127.0.0.1:11434",
        model="qwen2.5:7b-instruct-q4_0",
        timeout_seconds=5.0,
        transport=httpx.MockTransport(handler),
    )


def _generate(handler: Handler) -> object:
    async def run() -> object:
        return await generate_advice(
            _client(handler),
            decision=Decision.WHITE_SPOT_MARKER_DETECTED,
            guidance_item=_GUIDANCE_ITEM,
        )

    return asyncio.run(run())


def _respond(payload: dict[str, object]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": json.dumps(payload)})

    return handler


def test_a_well_formed_response_is_parsed() -> None:
    content = _generate(
        _respond(
            {
                "summary": "A marked region was found; it is not a diagnosis.",
                "immediate_actions": ["Retake the photo in even light.", "Isolate the pond log."],
                "prevention_actions": ["Keep consistent stocking density."],
                "additional_considerations": ["Ask a professional before acting further."],
            }
        )
    )
    assert content.summary.startswith("A marked region")
    assert content.immediate_actions == (
        "Retake the photo in even light.",
        "Isolate the pond log.",
    )
    assert content.prevention_actions == ("Keep consistent stocking density.",)
    assert content.additional_considerations == ("Ask a professional before acting further.",)


def test_extra_items_beyond_six_are_truncated_not_rejected() -> None:
    content = _generate(
        _respond(
            {
                "summary": "ok",
                "immediate_actions": [f"step {i}" for i in range(10)],
                "prevention_actions": ["one"],
            }
        )
    )
    assert len(content.immediate_actions) == 6


def test_additional_considerations_may_be_absent() -> None:
    content = _generate(
        _respond({"summary": "ok", "immediate_actions": ["a"], "prevention_actions": ["b"]})
    )
    assert content.additional_considerations == ()


@pytest.mark.parametrize(
    "payload",
    [
        {"immediate_actions": ["a"], "prevention_actions": ["b"]},
        {"summary": "", "immediate_actions": ["a"], "prevention_actions": ["b"]},
        {"summary": "ok", "immediate_actions": [], "prevention_actions": ["b"]},
        {"summary": "ok", "immediate_actions": ["a"], "prevention_actions": []},
        {"summary": "ok", "immediate_actions": "not a list", "prevention_actions": ["b"]},
    ],
)
def test_a_missing_or_empty_required_field_is_rejected(payload: dict[str, object]) -> None:
    with pytest.raises(AdviceGenerationError):
        _generate(_respond(payload))


def test_non_json_text_is_rejected() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "sure, here is your advice: ..."})

    with pytest.raises(AdviceGenerationError, match="valid JSON"):
        _generate(handler)


def test_a_prohibited_claim_in_the_summary_is_rejected() -> None:
    with pytest.raises(AdviceGenerationError, match="not entitled to make"):
        _generate(
            _respond(
                {
                    "summary": "The shrimp is healthy and the pond is safe to harvest.",
                    "immediate_actions": ["a"],
                    "prevention_actions": ["b"],
                }
            )
        )


def test_a_named_drug_in_an_action_is_rejected_even_as_a_warning() -> None:
    with pytest.raises(AdviceGenerationError, match="not entitled to make"):
        _generate(
            _respond(
                {
                    "summary": "ok",
                    "immediate_actions": ["Do not apply oxytetracycline without advice."],
                    "prevention_actions": ["b"],
                }
            )
        )


def test_a_dose_quantity_is_rejected() -> None:
    with pytest.raises(AdviceGenerationError, match="not entitled to make"):
        _generate(
            _respond(
                {
                    "summary": "ok",
                    "immediate_actions": ["a"],
                    "prevention_actions": ["Keep treatment below 5 mg/l at all times."],
                }
            )
        )


def test_an_unreachable_ollama_server_becomes_an_advice_generation_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(AdviceGenerationError):
        _generate(handler)
