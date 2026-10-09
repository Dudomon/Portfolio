"""Request-scoped access to models, enforcing the token budget on every call."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel, ValidationError

from rag_platform.agents.budget import TokenBudget
from rag_platform.clients.protocols import (
    ChatModel,
    ChatRequest,
    ChatResult,
    Embedder,
    EmbeddingResult,
    StreamCompleted,
    StreamEvent,
)
from rag_platform.domain import estimate_tokens


class AgentOutputError(Exception):
    """The model returned output that does not match the agent's expected schema."""


def _worst_case_tokens(request: ChatRequest) -> int:
    # Per-message overhead of a few tokens for role markers, plus the output ceiling.
    prompt = sum(estimate_tokens(m.content) + 4 for m in request.messages)
    return prompt + request.max_output_tokens


class ModelGateway:
    def __init__(
        self,
        chat: ChatModel,
        embedder: Embedder,
        budget: TokenBudget,
        embedding_deployment: str,
    ) -> None:
        self._chat = chat
        self._embedder = embedder
        self.budget = budget
        self._embedding_deployment = embedding_deployment

    async def complete(self, request: ChatRequest) -> ChatResult:
        with self.budget.reserve(_worst_case_tokens(request)) as reservation:
            result = await self._chat.complete(request)
            reservation.settle(request.operation, request.deployment, result.usage)
        return result

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        with self.budget.reserve(_worst_case_tokens(request)) as reservation:
            async for event in self._chat.stream(request):
                if isinstance(event, StreamCompleted):
                    reservation.settle(request.operation, request.deployment, event.result.usage)
                yield event

    async def complete_json[T: BaseModel](self, request: ChatRequest, schema: type[T]) -> T:
        result = await self.complete(request)
        try:
            return schema.model_validate(json.loads(result.text))
        except (json.JSONDecodeError, ValidationError) as error:
            raise AgentOutputError(f"{request.operation}: invalid structured output") from error

    async def embed(self, texts: Sequence[str], operation: str) -> EmbeddingResult:
        with self.budget.reserve(sum(estimate_tokens(t) for t in texts)) as reservation:
            result = await self._embedder.embed(texts)
            reservation.settle(operation, self._embedding_deployment, result.usage)
        return result
