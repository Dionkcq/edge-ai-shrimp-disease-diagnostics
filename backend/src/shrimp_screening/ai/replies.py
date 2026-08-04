"""Grounded fallbacks and domain checks for pond-side chat replies."""

from __future__ import annotations

from typing import Any


def _source_titles(guidance: dict[str, Any]) -> list[str]:
    sources = guidance.get("sources")
    if not isinstance(sources, list):
        return []
    titles: list[str] = []
    for source in sources:
        title = source.get("title") if isinstance(source, dict) else None
        if isinstance(title, str) and title.strip():
            titles.append(title.strip())
    return titles


def grounded_fallback(tool_result: dict[str, Any] | None) -> str:
    """Build a shrimp-specific answer without asking the model to improvise."""
    if not isinstance(tool_result, dict):
        return (
            "This assistant only supports shrimp image screening. Upload a shrimp photograph "
            "so the screening tool can check it before discussing what to do next."
        )

    if tool_result.get("status") == "needs_image":
        return (
            "Please upload a shrimp photograph first. I need a screening result before I can "
            "explain the next steps."
        )

    guidance = tool_result.get("guidance")
    if not isinstance(guidance, dict):
        return (
            "I cannot give a reliable next step because the screening result did not include "
            "the cited local shrimp guidance. Please retry the screen."
        )

    headline = guidance.get("headline")
    body = guidance.get("body")
    lead = (
        str(headline).strip()
        if isinstance(headline, str) and headline.strip()
        else "The shrimp image was screened."
    )
    action = str(body).strip() if isinstance(body, str) and body.strip() else ""
    sources = _source_titles(guidance)
    raw_review_note = guidance.get("review_note")
    review_note = (
        raw_review_note.strip()
        if isinstance(raw_review_note, str) and raw_review_note.strip()
        else ""
    )

    parts = [lead.rstrip(".") + "."]
    if action:
        parts.append(action)
    boundary = (
        "This appearance screen cannot identify the cause or recommend treatment or chemical "
        "dosing."
    )
    if "aquatic-animal health professional" not in action.lower():
        boundary += (
            " For decisions about the shrimp or pond, contact a qualified aquatic-animal "
            "health professional."
        )
    parts.append(boundary)
    if sources:
        label = "Source" if len(sources) == 1 else "Sources"
        parts.append(f"{label}: {', '.join(sources)}.")
    if review_note:
        parts.append(f"Guidance note: {review_note}")
    return " ".join(parts)


def guard_chat_reply(
    reply: str,
    tool_result: dict[str, Any] | None,
    *,
    user_message: str = "",
) -> str:
    """Never expose free-form model prose through the pond-side chat boundary."""
    del reply, user_message
    return grounded_fallback(tool_result)
