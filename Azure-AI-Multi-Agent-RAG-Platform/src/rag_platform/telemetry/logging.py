"""Structured JSON logging with trace correlation and PII redaction."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from opentelemetry import trace

from rag_platform.safety.redaction import redact

_STANDARD_ATTRIBUTES = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys() | {"message", "asctime"}
)


class _RedactingRecordFactory:
    """Wraps the active record factory so every record is redacted at creation time.

    Redacting here rather than in a handler filter matters: exporters such as Azure
    Monitor attach their own handlers, and logger-level filters do not apply to
    records propagated from child loggers. `extra=` fields are attached after the
    factory runs, so by convention they carry identifiers and counters only; the JSON
    formatter still redacts their string values as a second line of defense.
    """

    def __init__(self, base: Callable[..., logging.LogRecord]) -> None:
        self._base = base

    def __call__(self, *args: Any, **kwargs: Any) -> logging.LogRecord:
        record = self._base(*args, **kwargs)
        record.msg = redact(record.getMessage())
        record.args = None
        return record


class JsonFormatter(logging.Formatter):
    """One JSON object per line, enriched with the active trace and span ids."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            payload["trace_id"] = format(span_context.trace_id, "032x")
            payload["span_id"] = format(span_context.span_id, "016x")
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRIBUTES:
                payload[key] = redact(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str) -> None:
    current = logging.getLogRecordFactory()
    if not isinstance(current, _RedactingRecordFactory):
        logging.setLogRecordFactory(_RedactingRecordFactory(current))
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
