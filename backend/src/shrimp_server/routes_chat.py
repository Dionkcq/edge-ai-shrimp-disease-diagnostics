"""``POST /api/v1/chat`` -- the conversational agent boundary."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from starlette.datastructures import UploadFile
from starlette.requests import Request

from shrimp_screening.ai.agent import ScreeningAgent
from shrimp_screening.ai.tools import build_tool_registry
from shrimp_screening.contracts.chat import ChatResponse
from shrimp_screening.contracts.enums import ProblemCode
from shrimp_screening.llm.client import OllamaError
from shrimp_screening.problems import ApiProblemError
from shrimp_screening.resources import AppResources
from shrimp_server.dependencies import get_resources

router = APIRouter(prefix="/api/v1", tags=["chat"])
Resources = Annotated[AppResources, Depends(get_resources)]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, resources: Resources) -> ChatResponse:
    if resources.llm_client is None:
        raise ApiProblemError(
            ProblemCode.NOT_FOUND,
            404,
            "The local conversational agent is not enabled on this build.",
        )
    form = await request.form()
    message = str(form.get("message", "")).strip()
    session_value = form.get("session_id")
    session_id = str(session_value).strip() if session_value else None
    image_part = form.get("image")
    image: bytes | None = None
    if isinstance(image_part, UploadFile):
        image = await image_part.read(resources.settings.max_upload_bytes + 1)
        await image_part.close()
        if len(image) > resources.settings.max_upload_bytes:
            raise ApiProblemError(
                ProblemCode.PAYLOAD_TOO_LARGE,
                413,
                "The uploaded image is larger than the local limit.",
            )
    agent = ScreeningAgent(
        resources.llm_client,
        build_tool_registry(resources),
        resources.chat_memory,
    )
    try:
        reply, active_session, messages, tool_calls, tool_result = await agent.run(
            session_id=session_id,
            message=message,
            image=image,
        )
    except (OllamaError, ValueError) as exc:
        raise ApiProblemError(
            ProblemCode.ADVICE_UNAVAILABLE,
            503,
            "The local conversational agent could not complete this turn.",
            retry_after_seconds=resources.settings.retry_after_seconds,
        ) from exc
    return ChatResponse(
        session_id=active_session,
        reply=reply,
        messages=messages,
        tool_calls=tool_calls,
        tool_result=tool_result,
    )
