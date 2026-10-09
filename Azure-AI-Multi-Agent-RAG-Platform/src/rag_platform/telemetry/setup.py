"""Process-level telemetry bootstrap."""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider

from rag_platform.config import Settings
from rag_platform.telemetry.logging import configure_logging

logger = logging.getLogger(__name__)


def configure_telemetry(settings: Settings) -> None:
    """Configure logs and traces.

    With an Application Insights connection string, the Azure Monitor distro installs the
    tracer/meter/logger providers and auto-instruments FastAPI and httpx. Without one, a
    local SDK provider is still installed so trace ids are generated and returned to
    clients, which keeps log correlation working in development.
    """
    configure_logging(settings.log_level)
    resource = Resource.create({SERVICE_NAME: settings.service_name})
    connection_string = settings.applicationinsights_connection_string
    if connection_string is not None:
        # Imported lazily: the distro pulls in many instrumentation packages and is
        # only needed when exporting.
        from azure.monitor.opentelemetry import configure_azure_monitor  # noqa: PLC0415

        configure_azure_monitor(
            connection_string=connection_string.get_secret_value(),
            resource=resource,
            logger_name="rag_platform",
        )
        logger.info("telemetry configured", extra={"exporter": "azure_monitor"})
        return
    trace.set_tracer_provider(TracerProvider(resource=resource))
    logger.info("telemetry configured", extra={"exporter": "none"})
