"""Azure OpenAI adapters implementing the `ChatModel` and `Embedder` protocols."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import openai
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import get_bearer_token_provider
from openai import AsyncAzureOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import Choice

from rag_platform.clients.protocols import (
    ChatMessage,
    ChatRequest,
    ChatResult,
    EmbeddingResult,
    StreamCompleted,
    StreamEvent,
    TextDelta,
)
from rag_platform.config import Settings
from rag_platform.domain import TokenUsage
from rag_platform.safety.content_filter import (
    ContentFilteredError,
    filtered_categories,
    prompt_filter_error,
)

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def build_openai_client(
    settings: Settings, credential: AsyncTokenCredential | None
) -> AsyncAzureOpenAI:
    """Create the SDK client.

    An explicit API key wins (local development); otherwise Entra ID tokens are used,
    which in Azure come from the container's managed identity. The deployed account
    sets `disableLocalAuth`, so keys only ever work against developer resources.
    """
    common: dict[str, Any] = {
        "azure_endpoint": settings.azure_openai_endpoint,
        "api_version": settings.azure_openai_api_version,
        # Retries with backoff on 429/5xx are delegated to the SDK; the per-step
        # timeout in the orchestrator bounds the total time spent retrying.
        "max_retries": 3,
    }
    if settings.azure_openai_api_key is not None:
        return AsyncAzureOpenAI(api_key=settings.azure_openai_api_key.get_secret_value(), **common)
    if credential is None:
        raise ValueError("A token credential is required when no Azure OpenAI API key is set")
    token_provider = get_bearer_token_provider(credential, COGNITIVE_SERVICES_SCOPE)
    return AsyncAzureOpenAI(azure_ad_token_provider=token_provider, **common)


def _to_sdk_message(message: ChatMessage) -> ChatCompletionMessageParam:
    match message.role:
        case "system":
            return ChatCompletionSystemMessageParam(role="system", content=message.content)
        case "user":
            return ChatCompletionUserMessageParam(role="user", content=message.content)
        case "assistant":
            return ChatCompletionAssistantMessageParam(role="assistant", content=message.content)


def _completion_filter(extra: Mapping[str, Any] | None) -> ContentFilteredError:
    results = (extra or {}).get("content_filter_results")
    return ContentFilteredError(
        "completion", filtered_categories(results if isinstance(results, Mapping) else None)
    )


class AzureChatModel:
    def __init__(self, client: AsyncAzureOpenAI) -> None:
        self._client = client

    @staticmethod
    def _arguments(request: ChatRequest) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "model": request.deployment,
            "messages": [_to_sdk_message(m) for m in request.messages],
            "max_completion_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        }
        if request.json_output:
            arguments["response_format"] = {"type": "json_object"}
        return arguments

    async def complete(self, request: ChatRequest) -> ChatResult:
        try:
            response = await self._client.chat.completions.create(**self._arguments(request))
        except openai.BadRequestError as error:
            filtered = prompt_filter_error(error.body)
            if filtered is not None:
                raise filtered from error
            raise
        choice: Choice = response.choices[0]
        if choice.finish_reason == "content_filter":
            raise _completion_filter(choice.model_extra)
        usage = response.usage
        return ChatResult(
            text=choice.message.content or "",
            usage=TokenUsage(
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            ),
            model=response.model,
            finish_reason=choice.finish_reason,
        )

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        try:
            stream = await self._client.chat.completions.create(
                **self._arguments(request),
                stream=True,
                stream_options={"include_usage": True},
            )
        except openai.BadRequestError as error:
            filtered = prompt_filter_error(error.body)
            if filtered is not None:
                raise filtered from error
            raise
        parts: list[str] = []
        usage = TokenUsage()
        model = request.deployment
        finish_reason: str | None = None
        # `async with` closes the HTTP response if the consumer stops early or is canceled.
        async with stream:
            async for chunk in stream:
                model = chunk.model or model
                if chunk.usage is not None:
                    usage = TokenUsage(
                        input_tokens=chunk.usage.prompt_tokens,
                        output_tokens=chunk.usage.completion_tokens,
                    )
                # Azure emits a first chunk with no choices that carries prompt filter results.
                if not chunk.choices:
                    continue
                choice = chunk.choices[0]
                if choice.finish_reason == "content_filter":
                    raise _completion_filter(choice.model_extra)
                finish_reason = choice.finish_reason or finish_reason
                if choice.delta.content:
                    parts.append(choice.delta.content)
                    yield TextDelta(text=choice.delta.content)
        yield StreamCompleted(
            result=ChatResult(
                text="".join(parts), usage=usage, model=model, finish_reason=finish_reason
            )
        )


class AzureEmbedder:
    def __init__(self, client: AsyncAzureOpenAI, deployment: str, dimensions: int) -> None:
        self._client = client
        self._deployment = deployment
        self._dimensions = dimensions

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        response = await self._client.embeddings.create(
            model=self._deployment, input=list(texts), dimensions=self._dimensions
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return EmbeddingResult(
            vectors=[item.embedding for item in ordered],
            usage=TokenUsage(input_tokens=response.usage.prompt_tokens),
        )
