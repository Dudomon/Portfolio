from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx
import pytest

from rag_platform.agents.orchestrator import Orchestrator
from rag_platform.api import create_app
from rag_platform.config import Settings

from conftest import POLICY_HITS, FakeChatModel, FakeEmbedder, Reply, StaticSearch


@pytest.fixture
async def client(
    settings: Settings, replies: dict[str, list[Reply]]
) -> AsyncIterator[httpx.AsyncClient]:
    orchestrator = Orchestrator(
        FakeChatModel(replies), FakeEmbedder(), StaticSearch(POLICY_HITS), settings
    )
    app = create_app(orchestrator=orchestrator, settings=settings)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


def parse_sse(body: str) -> list[tuple[str, dict[str, object]]]:
    events = []
    for block in body.strip().split("\n\n"):
        fields = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((fields["event"], json.loads(fields["data"])))
    return events


async def test_healthz(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_answer_returns_result_with_citations(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/answers", json={"question": "How many quotes for $30k?"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "answered"
    assert body["grounded"] is True
    assert body["citations"][0]["doc_id"] == "PROC-001"
    assert body["citations"][0]["source_number"] == 1
    assert body["cost_usd"] > 0


@pytest.mark.parametrize(
    "payload", [{"question": "hi"}, {"question": "x" * 2001}, {"q": "missing field"}]
)
async def test_invalid_requests_are_rejected(
    client: httpx.AsyncClient, payload: dict[str, str]
) -> None:
    response = await client.post("/v1/answers", json=payload)

    assert response.status_code == 422


async def test_stream_emits_typed_events_ending_with_completed(client: httpx.AsyncClient) -> None:
    response = await client.post("/v1/answers/stream", json={"question": "How many quotes?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(response.text)
    names = [name for name, _ in events]
    assert names[:2] == ["plan", "retrieval"]
    assert "answer_delta" in names
    assert names[-2:] == ["verification", "completed"]
    streamed = "".join(str(data["text"]) for name, data in events if name == "answer_delta")
    completed = events[-1][1]["result"]
    assert isinstance(completed, dict)
    assert streamed == completed["answer"]


async def test_stream_reports_unexpected_failures_in_band(
    settings: Settings, replies: dict[str, list[Reply]]
) -> None:
    replies["planner"] = [RuntimeError("boom")]
    orchestrator = Orchestrator(
        FakeChatModel(replies), FakeEmbedder(), StaticSearch(POLICY_HITS), settings
    )
    transport = httpx.ASGITransport(app=create_app(orchestrator=orchestrator, settings=settings))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        response = await http.post("/v1/answers/stream", json={"question": "How many quotes?"})

    (event,) = parse_sse(response.text)
    assert event[0] == "error"
    assert "boom" not in response.text
