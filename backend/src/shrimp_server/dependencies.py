"""Reaching the process-wide state from inside a request.

The state itself -- what it contains and how it is built -- is
:mod:`shrimp_screening.resources`, which knows nothing about HTTP. All that lives
here is the ASGI-specific part: where the bundle is stashed on the application and
how a route asks for it.
"""

from __future__ import annotations

from typing import cast

from starlette.requests import Request

from shrimp_screening.resources import AppResources

#: Key under which the resource bundle is stored on the ASGI app state.
RESOURCES_ATTRIBUTE = "resources"


def get_resources(request: Request) -> AppResources:
    return cast(AppResources, getattr(request.app.state, RESOURCES_ATTRIBUTE))


def get_request_id(request: Request) -> str:
    return cast(str, request.state.request_id)
