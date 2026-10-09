"""Exercises the real `openai` SDK against a mocked HTTP transport (no network, no Azure).

The SDK (3.x) is built on `httpx2`, so its mock transport is used here.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx2
import pytest
from openai import AsyncAzureOpenAI

from rag_platform.clients.azure_openai import AzureChatModel, AzureEmbedder
from rag_platform.clients.protocols import ChatMessage, ChatRequest, StreamCompleted, TextDelta
from rag_platform.safety.content_filter import ContentFilteredError

REQUEST = ChatRequest(
    deployment="gpt-4o",
    operation="answer",
    max_output_tokens=50,
    json_output=True,
    messages=(ChatMessage(role="system", content="s"), ChatMessage(role="user", content="q")),
)


def client_for(handler: Callable[[httpx2.Request], httpx2.Response]) -> AsyncAzureOpenAI:
    return AsyncAzureOpenAI(
        azure_endpoint="https://unit-test.openai.azure.com",
        api_key="test-key",
        api_version="2024-10-21",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)),
    )


def completion(finish_reason: str, content: str | None, **choice_extra: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": content},
                **choice_extra,
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    }


async def test_complete_maps_response_and_sends_expected_request() -> None:
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(200, json=completion("stop", '{"ok": true}'))

    result = await AzureChatModel(client_for(handler)).complete(REQUEST)

    assert result.text == '{"ok": true}'
    assert (result.usage.input_tokens, result.usage.output_tokens) == (12, 3)
    assert result.model == "gpt-4o-2024-08-06"
    (request,) = seen
    assert request.url.path == "/openai/deployments/gpt-4o/chat/completions"
    assert request.url.params["api-version"] == "2024-10-21"
    body = json.loads(request.content)
    assert body["response_format"] == {"type": "json_object"}
    assert body["max_completion_tokens"] == 50


async def test_completion_content_filter_raises() -> None:
    filtered = {"violence": {"filtered": True, "severity": "high"}}

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, json=completion("content_filter", None, content_filter_results=filtered)
        )

    with pytest.raises(ContentFilteredError) as raised:
        await AzureChatModel(client_for(handler)).complete(REQUEST)

    assert raised.value.stage == "completion"
    assert raised.value.categories == ("violence",)


async def test_prompt_content_filter_raises() -> None:
    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            400,
            json={
                "error": {
                    "message": "The response was filtered",
                    "code": "content_filter",
                    "status": 400,
                    "innererror": {
                        "code": "ResponsibleAIPolicyViolation",
                        "content_filter_result": {
                            "jailbreak": {"filtered": True, "detected": True}
                        },
                    },
                }
            },
        )

    with pytest.raises(ContentFilteredError) as raised:
        await AzureChatModel(client_for(handler)).complete(REQUEST)

    assert raised.value.stage == "prompt"
    assert raised.value.categories == ("jailbreak",)


async def test_stream_yields_deltas_then_usage() -> None:
    def sse(payload: dict[str, Any]) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    base = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "gpt-4o"}
    body = (
        sse({**base, "choices": [], "prompt_filter_results": []})
        + sse({**base, "choices": [{"index": 0, "delta": {"content": "Three"}}]})
        + sse({**base, "choices": [{"index": 0, "delta": {"content": " quotes"}}]})
        + sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        + sse(
            {
                **base,
                "choices": [],
                "usage": {"prompt_tokens": 20, "completion_tokens": 2, "total_tokens": 22},
            }
        )
        + "data: [DONE]\n\n"
    )

    def handler(_: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    events = [e async for e in AzureChatModel(client_for(handler)).stream(REQUEST)]

    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Three", " quotes"]
    final = events[-1]
    assert isinstance(final, StreamCompleted)
    assert final.result.text == "Three quotes"
    assert final.result.usage.input_tokens == 20
    assert final.result.finish_reason == "stop"


async def test_embedder_orders_vectors_by_index() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert json.loads(request.content)["dimensions"] == 3
        return httpx2.Response(
            200,
            json={
                "object": "list",
                "model": "text-embedding-3-large",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0, 0.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 8, "total_tokens": 8},
            },
        )

    embedder = AzureEmbedder(client_for(handler), "text-embedding-3-large", dimensions=3)
    result = await embedder.embed(["first", "second"])

    assert result.vectors == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    assert result.usage.input_tokens == 8
