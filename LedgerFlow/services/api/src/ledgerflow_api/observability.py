"""Structured logging, Prometheus metrics and optional OpenTelemetry tracing."""

import logging
import sys

import structlog
from fastapi import FastAPI
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import Counter, Histogram
from sqlalchemy.ext.asyncio import AsyncEngine

from ledgerflow_api.config import Settings

HTTP_REQUESTS = Counter(
    "ledgerflow_http_requests_total",
    "HTTP requests handled, by route template and status code.",
    ["method", "route", "status"],
)
HTTP_LATENCY = Histogram(
    "ledgerflow_http_request_duration_seconds",
    "HTTP request latency by route template.",
    ["method", "route"],
)


def configure_logging(settings: Settings) -> None:
    """Route both structlog and stdlib logging (uvicorn, sqlalchemy) through one JSON renderer."""
    shared: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                structlog.processors.format_exc_info,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(settings.log_level.upper())
    # The access log middleware already records every request with more context.
    logging.getLogger("uvicorn.access").disabled = True


def configure_tracing(settings: Settings, app: FastAPI) -> TracerProvider | None:
    """Enable tracing only when an OTLP endpoint is configured; otherwise this is a no-op.

    Must run before the app starts serving: instrumentation installs ASGI middleware.
    """
    if not settings.otel_exporter_otlp_endpoint:
        return None
    provider = TracerProvider(
        resource=Resource.create({"service.name": settings.otel_service_name})
    )
    provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{settings.otel_exporter_otlp_endpoint}/v1/traces")
        )
    )
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(
        app, tracer_provider=provider, excluded_urls="healthz,readyz,metrics"
    )
    return provider


def instrument_engine(engine: AsyncEngine, provider: TracerProvider) -> None:
    SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine, tracer_provider=provider)
