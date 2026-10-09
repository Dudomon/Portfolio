"""OpenTelemetry instrumentation for model calls, following the GenAI semantic conventions.

Instrumentation is a decorator over the `ChatModel` / `Embedder` protocols rather than
code inside the Azure adapters, so the exact same spans are produced in tests (with
fakes) and in production, and the span contract can be asserted in unit tests.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, SpanKind, Status, StatusCode

from rag_platform.clients.protocols import (
    ChatModel,
    ChatRequest,
    ChatResult,
    Embedder,
    EmbeddingResult,
    StreamCompleted,
    StreamEvent,
)
from rag_platform.domain import TokenUsage
from rag_platform.safety.content_filter import ContentFilteredError

# `gen_ai.provider.name` is the current convention; `gen_ai.system` is kept because
# Application Insights' agent and token views still key on it.
PROVIDER_ATTRIBUTES = {"gen_ai.provider.name": "azure.ai.openai", "gen_ai.system": "az.ai.openai"}

_tracer = trace.get_tracer("rag_platform.genai")
_meter = metrics.get_meter("rag_platform.genai")
_duration = _meter.create_histogram(
    "gen_ai.client.operation.duration", unit="s", description="GenAI operation duration"
)
_token_usage = _meter.create_histogram(
    "gen_ai.client.token.usage", unit="{token}", description="Tokens used per GenAI operation"
)


def _record_usage(span: Span, usage: TokenUsage, metric_attributes: dict[str, str]) -> None:
    span.set_attribute("gen_ai.usage.input_tokens", usage.input_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", usage.output_tokens)
    _token_usage.record(usage.input_tokens, {**metric_attributes, "gen_ai.token.type": "input"})
    _token_usage.record(usage.output_tokens, {**metric_attributes, "gen_ai.token.type": "output"})


def _record_error(span: Span, error: Exception) -> None:
    span.set_attribute("error.type", type(error).__name__)
    if isinstance(error, ContentFilteredError):
        span.set_attribute("rag.content_filter.stage", error.stage)
        span.set_attribute("rag.content_filter.categories", list(error.categories))
    span.set_status(Status(StatusCode.ERROR, str(error)))


class InstrumentedChatModel:
    """Wraps any `ChatModel`, emitting one CLIENT span per call plus duration/token metrics."""

    def __init__(self, inner: ChatModel) -> None:
        self._inner = inner

    @staticmethod
    def _start_span(request: ChatRequest) -> Span:
        span = _tracer.start_span(f"chat {request.deployment}", kind=SpanKind.CLIENT)
        span.set_attributes(
            {
                **PROVIDER_ATTRIBUTES,
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": request.deployment,
                "gen_ai.request.max_tokens": request.max_output_tokens,
                "gen_ai.request.temperature": request.temperature,
                "rag.operation": request.operation,
            }
        )
        return span

    @staticmethod
    def _finish(span: Span, request: ChatRequest, result: ChatResult, started: float) -> None:
        span.set_attribute("gen_ai.response.model", result.model)
        if result.finish_reason:
            span.set_attribute("gen_ai.response.finish_reasons", [result.finish_reason])
        attributes = {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": request.deployment,
            "rag.operation": request.operation,
        }
        _record_usage(span, result.usage, attributes)
        _duration.record(time.perf_counter() - started, attributes)

    async def complete(self, request: ChatRequest) -> ChatResult:
        span = self._start_span(request)
        started = time.perf_counter()
        with trace.use_span(
            span, end_on_exit=True, record_exception=False, set_status_on_exception=False
        ):
            try:
                result = await self._inner.complete(request)
            except Exception as error:
                _record_error(span, error)
                raise
            self._finish(span, request, result, started)
            return result

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        # The span is not made "current" here: attaching a context inside an async
        # generator leaks it into the consumer between yields and breaks detaching.
        span = self._start_span(request)
        started = time.perf_counter()
        first_token_at: float | None = None
        try:
            async for event in self._inner.stream(request):
                if isinstance(event, StreamCompleted):
                    self._finish(span, request, event.result, started)
                elif first_token_at is None:
                    first_token_at = time.perf_counter()
                    span.set_attribute(
                        "rag.time_to_first_token_ms", round((first_token_at - started) * 1000, 1)
                    )
                yield event
        except Exception as error:
            _record_error(span, error)
            raise
        finally:
            span.end()


class InstrumentedEmbedder:
    def __init__(self, inner: Embedder, deployment: str) -> None:
        self._inner = inner
        self._deployment = deployment

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        attributes = {
            "gen_ai.operation.name": "embeddings",
            "gen_ai.request.model": self._deployment,
        }
        started = time.perf_counter()
        with _tracer.start_as_current_span(
            f"embeddings {self._deployment}", kind=SpanKind.CLIENT
        ) as span:
            span.set_attributes({**PROVIDER_ATTRIBUTES, **attributes, "rag.batch_size": len(texts)})
            result = await self._inner.embed(texts)
            _record_usage(span, result.usage, attributes)
            _duration.record(time.perf_counter() - started, attributes)
            return result
