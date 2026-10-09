"""Shared fixtures and in-memory fakes for the model and search protocols."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from pathlib import Path

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import SecretStr

from rag_platform.clients.protocols import (
    ChatRequest,
    ChatResult,
    EmbeddingResult,
    SearchQuery,
    StreamCompleted,
    StreamEvent,
    TextDelta,
)
from rag_platform.config import Settings
from rag_platform.domain import RetrievedChunk, TokenUsage, estimate_tokens

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The global tracer provider can be set only once per process.
_SPAN_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_SPAN_EXPORTER))
trace.set_tracer_provider(_provider)

Reply = str | Exception | Callable[[ChatRequest], str]


class FakeChatModel:
    """Scripted chat model: replies are queued per `operation` and consumed in order.

    The last reply of an operation is reused when the queue runs out, which keeps
    tests short when an agent is called several times with the same answer.
    """

    def __init__(self, replies: dict[str, list[Reply]], delay_seconds: float = 0.0) -> None:
        self._replies = {op: list(items) for op, items in replies.items()}
        self.delay_seconds = delay_seconds
        self.requests: list[ChatRequest] = []
        self.canceled = 0

    def calls(self, operation: str) -> list[ChatRequest]:
        return [r for r in self.requests if r.operation == operation]

    async def _reply(self, request: ChatRequest) -> str:
        self.requests.append(request)
        try:
            await asyncio.sleep(self.delay_seconds)
        except asyncio.CancelledError:
            self.canceled += 1
            raise
        queue = self._replies.get(request.operation)
        if not queue:
            raise AssertionError(f"No scripted reply for operation {request.operation!r}")
        reply = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(reply, Exception):
            raise reply
        return reply(request) if callable(reply) else reply

    @staticmethod
    def _result(request: ChatRequest, text: str) -> ChatResult:
        prompt = sum(estimate_tokens(m.content) for m in request.messages)
        return ChatResult(
            text=text,
            usage=TokenUsage(input_tokens=prompt, output_tokens=estimate_tokens(text)),
            model=f"{request.deployment}-2024-08-06",
            finish_reason="stop",
        )

    async def complete(self, request: ChatRequest) -> ChatResult:
        return self._result(request, await self._reply(request))

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        text = await self._reply(request)
        for index, word in enumerate(text.split(" ")):
            yield TextDelta(text=word if index == 0 else f" {word}")
        yield StreamCompleted(result=self._result(request, text))


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        self.calls.append(list(texts))
        vectors = [[float(len(t)), float(sum(map(ord, t)) % 97), 1.0] for t in texts]
        return EmbeddingResult(
            vectors=vectors, usage=TokenUsage(input_tokens=sum(estimate_tokens(t) for t in texts))
        )


class StaticSearch:
    """Returns the same hits for every query and records the queries."""

    def __init__(self, hits: Sequence[RetrievedChunk]) -> None:
        self.hits = list(hits)
        self.queries: list[SearchQuery] = []

    async def hybrid_search(self, query: SearchQuery) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.hits[: query.top]


def chunk(
    chunk_id: str, doc_id: str, content: str, reranker_score: float | None = 2.5
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=f"{doc_id} policy",
        heading_path=f"{doc_id} policy > Section",
        content=content,
        source_path=f"{doc_id.lower()}.md",
        score=0.03,
        reranker_score=reranker_score,
    )


POLICY_HITS = (
    chunk("PROC-001-a", "PROC-001", "Orders above USD 25,000 require at least three quotes."),
    chunk("PROC-001-b", "PROC-001", "Orders up to 50,000 need the Finance Controller."),
    chunk("TAX-002-a", "TAX-002", "The default WHT rate is 30 percent of the gross payment."),
)


def critic_reply(*claims: tuple[str, list[int], bool]) -> str:
    return json.dumps(
        {
            "claims": [
                {"claim": text, "sources": sources, "supported": ok, "reason": "checked"}
                for text, sources, ok in claims
            ]
        }
    )


def default_replies() -> dict[str, list[Reply]]:
    return {
        "planner": [json.dumps({"sub_questions": ["How many quotes are needed above $25k?"]})],
        "query_rewrite": [
            json.dumps({"keyword_query": "quotes competitive bidding", "semantic_query": "quotes?"})
        ],
        "answer": ["You need three written quotes for orders above USD 25,000 [1]."],
        "critic": [critic_reply(("Three quotes above USD 25,000", [1], True))],
    }


@pytest.fixture
def settings() -> Settings:
    return Settings(
        _env_file=None,
        azure_openai_api_key=SecretStr("test-key"),
        azure_search_api_key=SecretStr("test-key"),
        request_token_budget=20_000,
        request_timeout_seconds=5,
        step_timeout_seconds=2,
    )


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    _SPAN_EXPORTER.clear()
    yield _SPAN_EXPORTER
    _SPAN_EXPORTER.clear()


def span_named(exporter: InMemorySpanExporter, name: str) -> list[ReadableSpan]:
    return [s for s in exporter.get_finished_spans() if s.name == name]


@pytest.fixture
def corpus_dir() -> Path:
    return PROJECT_ROOT / "data" / "sample_corpus"


@pytest.fixture
def replies() -> dict[str, list[Reply]]:
    return default_replies()
