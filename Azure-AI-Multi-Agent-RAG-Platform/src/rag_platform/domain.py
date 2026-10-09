"""Core value types shared by ingestion, retrieval, agents and evaluation."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceDocument(_Frozen):
    """A loaded document before chunking."""

    doc_id: str
    title: str
    source_path: str
    text: str
    category: str = "general"
    owner: str | None = None
    effective_date: date | None = None


class Chunk(_Frozen):
    """A retrievable unit of a document, ready to be embedded and indexed."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    title: str
    heading_path: tuple[str, ...]
    content: str
    source_path: str
    category: str
    content_hash: str
    token_estimate: int

    @property
    def heading_label(self) -> str:
        return " > ".join(self.heading_path)


class RetrievedChunk(_Frozen):
    """A chunk returned by the search backend, with its relevance scores."""

    chunk_id: str
    doc_id: str
    title: str
    heading_path: str
    content: str
    source_path: str
    score: float
    reranker_score: float | None = None


class TokenUsage(_Frozen):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


def estimate_tokens(text: str) -> int:
    """Cheap, conservative token estimate used for chunk sizing and budget reservations.

    Roughly four characters per token holds for English prose with GPT tokenizers.
    It deliberately avoids a tokenizer dependency (and its runtime vocabulary
    download); authoritative counts always come from the API usage payload.
    """
    return max(1, -(-len(text) // 4))
