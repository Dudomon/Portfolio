from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from sqlalchemy import text

from ledgerflow_api.api.deps import AppResources, Resources
from ledgerflow_api.api.v1 import api_router
from ledgerflow_api.config import Settings, get_settings
from ledgerflow_api.db.session import create_engine, create_session_factory, system_session
from ledgerflow_api.errors import register_exception_handlers
from ledgerflow_api.middleware import RequestContextMiddleware
from ledgerflow_api.observability import configure_logging, configure_tracing, instrument_engine
from ledgerflow_api.security import TokenService
from ledgerflow_api.services.storage import S3Storage

logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    tracer_provider: TracerProvider | None = app.state.tracer_provider
    engine = create_engine(settings)
    if tracer_provider is not None:
        instrument_engine(engine, tracer_provider)
    try:
        async with S3Storage.connect(settings) as storage:
            app.state.resources = AppResources(
                settings=settings,
                engine=engine,
                session_factory=create_session_factory(engine),
                tokens=TokenService(settings),
                storage=storage,
            )
            logger.info("api_started", environment=settings.environment)
            yield
    finally:
        await engine.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings)

    app = FastAPI(
        title="LedgerFlow API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment != "production" else None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.tracer_provider = configure_tracing(settings, app)
    register_exception_handlers(app)

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_methods=["GET", "POST", "PATCH"],
            allow_headers=["Authorization", "Content-Type", "If-Match", "X-Request-ID"],
            expose_headers=["ETag", "X-Request-ID"],
        )
    # Added after CORS so it wraps it: preflight responses get a request id as well.
    app.add_middleware(RequestContextMiddleware)

    app.include_router(api_router)

    @app.get("/healthz", include_in_schema=False)
    async def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readiness(resources: Resources) -> dict[str, str]:
        async with system_session(resources.session_factory) as session:
            await session.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app
