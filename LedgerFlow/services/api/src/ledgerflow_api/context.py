"""Ambient per-request context shared by middleware, services and error handlers."""

from contextvars import ContextVar

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
