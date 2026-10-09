"""HTTP API: non-streaming and Server-Sent Events answer endpoints."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from rag_platform import __version__
from rag_platform.agents.messages import AgentEvent, AnswerResult
from rag_platform.agents.orchestrator import Orchestrator
from rag_platform.config import Settings, get_settings
from rag_platform.telemetry.setup import configure_telemetry
from rag_platform.wiring import azure_runtime

logger = logging.getLogger(__name__)


class AnswerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=3, max_length=2000)


def _sse(event_type: str, payload: str) -> str:
    return f"event: {event_type}\ndata: {payload}\n\n"


async def _event_stream(events: AsyncIterator[AgentEvent]) -> AsyncIterator[str]:
    try:
        async for event in events:
            yield _sse(event.type, event.model_dump_json())
    except Exception:
        # Headers are already sent, so the failure is reported in-band; details stay
        # in the logs (with the trace id) rather than leaking to the client.
        logger.exception("streaming request failed")
        yield _sse("error", '{"message": "The request failed. See service logs."}')


def get_orchestrator(request: Request) -> Orchestrator:
    orchestrator: Orchestrator = request.app.state.orchestrator
    return orchestrator


OrchestratorDependency = Annotated[Orchestrator, Depends(get_orchestrator)]


def create_app(
    orchestrator: Orchestrator | None = None, settings: Settings | None = None
) -> FastAPI:
    """Application factory.

    Tests inject an orchestrator wired with fakes; in production the lifespan builds
    the Azure clients once per process and closes them on shutdown.
    """
    resolved_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if orchestrator is not None:
            yield
            return
        configure_telemetry(resolved_settings)
        async with azure_runtime(resolved_settings) as runtime:
            app.state.orchestrator = Orchestrator(
                runtime.chat, runtime.embedder, runtime.search, resolved_settings
            )
            yield

    app = FastAPI(
        title="Policy Knowledge Assistant",
        version=__version__,
        lifespan=lifespan,
    )
    if orchestrator is not None:
        # Set eagerly rather than in the lifespan so in-process ASGI test clients,
        # which do not send lifespan events, see it too.
        app.state.orchestrator = orchestrator

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/answers", tags=["answers"])
    async def answer(body: AnswerRequest, orchestrator: OrchestratorDependency) -> AnswerResult:
        # Outcomes such as "insufficient_evidence" or "budget_exhausted" are valid
        # results with a 200 status; clients branch on `status`, not on HTTP codes.
        return await orchestrator.answer(body.question)

    @app.post(
        "/v1/answers/stream",
        tags=["answers"],
        response_class=StreamingResponse,
        responses={200: {"content": {"text/event-stream": {}}}},
    )
    async def answer_stream(
        body: AnswerRequest, orchestrator: OrchestratorDependency
    ) -> StreamingResponse:
        return StreamingResponse(
            _event_stream(orchestrator.stream(body.question)),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
