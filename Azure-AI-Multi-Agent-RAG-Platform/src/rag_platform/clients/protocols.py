"""Narrow interfaces over the Azure SDKs.

The orchestration layer depends only on these protocols, which keeps the agents
testable with in-memory fakes and makes the Azure-specific code a thin adapter.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from rag_platform.domain import RetrievedChunk, TokenUsage

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Role
    content: str


class ChatRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    deployment: str
    messages: tuple[ChatMessage, ...]
    max_output_tokens: int = Field(gt=0)
    temperature: float = 0.0
    json_output: bool = False
    # Logical operation name, used for span names and cost attribution.
    operation: str


class ChatResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    usage: TokenUsage
    model: str
    finish_reason: str | None = None


class TextDelta(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str


class StreamCompleted(BaseModel):
    model_config = ConfigDict(frozen=True)

    result: ChatResult


StreamEvent = TextDelta | StreamCompleted


class ChatModel(Protocol):
    async def complete(self, request: ChatRequest) -> ChatResult: ...

    def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        """Yield text deltas, then exactly one `StreamCompleted` carrying usage."""
        ...


class EmbeddingResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    vectors: list[list[float]]
    usage: TokenUsage


class Embedder(Protocol):
    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...


class SearchQuery(BaseModel):
    """A hybrid query: BM25 over `keyword_text`; vector and semantic ranking use `semantic_text`."""

    model_config = ConfigDict(frozen=True)

    keyword_text: str
    semantic_text: str
    vector: tuple[float, ...]
    top: int = Field(gt=0)
    odata_filter: str | None = None


class SearchBackend(Protocol):
    async def hybrid_search(self, query: SearchQuery) -> list[RetrievedChunk]: ...


class IndexStore(Protocol):
    """Write-side operations the ingestion pipeline needs from the search index."""

    async def ensure_index(self) -> None: ...

    async def chunk_hashes(self, doc_id: str) -> dict[str, str]:
        """Return `chunk_id -> content_hash` for every chunk currently indexed for `doc_id`."""
        ...

    async def upsert(self, documents: Sequence[Mapping[str, Any]]) -> None: ...

    async def delete(self, chunk_ids: Sequence[str]) -> None: ...
