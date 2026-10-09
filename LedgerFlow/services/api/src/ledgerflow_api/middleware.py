"""Request id propagation, access logging and HTTP metrics as a single pure-ASGI middleware.

Pure ASGI rather than ``BaseHTTPMiddleware`` so streaming responses aren't buffered and
contextvars set here are visible to the endpoint.
"""

import re
import time
import uuid

import structlog
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from ledgerflow_api.context import request_id_var
from ledgerflow_api.observability import HTTP_LATENCY, HTTP_REQUESTS

REQUEST_ID_HEADER = "X-Request-ID"
# Accept a caller's id only if it is safe to echo into logs and headers.
_VALID_REQUEST_ID = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")
_UNMATCHED_ROUTE = "<unmatched>"
_QUIET_PATHS = frozenset({"/healthz", "/readyz", "/metrics"})

access_logger = structlog.get_logger("ledgerflow.access")


def _incoming_request_id(scope: Scope) -> str:
    for name, value in scope["headers"]:
        if name == b"x-request-id":
            candidate = bytes(value).decode("latin-1")
            if _VALID_REQUEST_ID.fullmatch(candidate):
                return candidate
    return uuid.uuid4().hex


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = _incoming_request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id
        token = request_id_var.set(request_id)
        structlog.contextvars.bind_contextvars(request_id=request_id)
        status_code = 500
        started = time.perf_counter()

        async def send_with_request_id(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                MutableHeaders(scope=message)[REQUEST_ID_HEADER] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_with_request_id)
        finally:
            elapsed = time.perf_counter() - started
            # The route template (not the raw path) keeps metric label cardinality bounded.
            route = getattr(scope.get("route"), "path", _UNMATCHED_ROUTE)
            method = scope["method"]
            HTTP_REQUESTS.labels(method=method, route=route, status=str(status_code)).inc()
            HTTP_LATENCY.labels(method=method, route=route).observe(elapsed)
            if scope["path"] not in _QUIET_PATHS:
                access_logger.info(
                    "http_request",
                    method=method,
                    path=scope["path"],
                    route=route,
                    status=status_code,
                    duration_ms=round(elapsed * 1000, 2),
                )
            structlog.contextvars.unbind_contextvars("request_id")
            request_id_var.reset(token)
