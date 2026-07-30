"""Per-request identity and response hardening.

Implemented as raw ASGI rather than ``BaseHTTPMiddleware`` for one concrete
reason: the screening route consumes ``request.stream()`` directly so it can abort
an oversized upload mid-flight, and wrapping the receive channel in an extra
task-and-queue layer is exactly the kind of thing that quietly breaks that.

The Content-Security-Policy is not decoration. ``default-src 'self'`` with no CDN
origins is the mechanical proof of the project's offline claim: if a built asset
ever referenced a remote host, the browser would block it and the demonstration
would fail loudly instead of silently depending on a network that a pond does not
have.
"""

from __future__ import annotations

from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from shrimp_screening.ids import new_ulid

CONTENT_SECURITY_POLICY = (
    "default-src 'self'; "
    "img-src 'self' blob: data:; "
    "connect-src 'self'; "
    "style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; "
    "object-src 'none'; "
    "base-uri 'none'; "
    "form-action 'self'; "
    "frame-ancestors 'none'"
)

SECURITY_HEADERS: tuple[tuple[str, str], ...] = (
    ("X-Content-Type-Options", "nosniff"),
    ("Referrer-Policy", "no-referrer"),
    ("Content-Security-Policy", CONTENT_SECURITY_POLICY),
    ("Cross-Origin-Opener-Policy", "same-origin"),
    ("Cross-Origin-Resource-Policy", "same-origin"),
    ("Permissions-Policy", "geolocation=(), microphone=(), interest-cohort=()"),
)


class RequestContextMiddleware:
    """Assigns a ULID to every request and stamps the security headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = new_ulid()
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id
        is_api = scope.get("path", "").startswith("/api/")

        async def send_with_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS:
                    headers.setdefault(name, value)
                headers.setdefault("X-Request-ID", request_id)
                # A screening result describes one ephemeral photograph; caching it
                # anywhere would outlive the image it came from.
                headers.setdefault("Cache-Control", "no-store" if is_api else "no-cache")
            await send(message)

        await self.app(scope, receive, send_with_headers)
