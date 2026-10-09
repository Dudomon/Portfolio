"""Error rendering through the real app. None of these paths reach the database."""

from collections.abc import AsyncIterator

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from ledgerflow_api.main import create_app
from tests.conftest import make_settings

PROBLEM_KEYS = {"type", "title", "status", "detail", "instance", "request_id"}


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app(make_settings())
    async with (
        LifespanManager(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as http,
    ):
        yield http


async def test_unauthenticated_request_is_problem_json(client: AsyncClient) -> None:
    response = await client.get("/api/v1/me")

    assert response.status_code == 401
    assert response.headers["content-type"] == "application/problem+json"
    assert response.headers["www-authenticate"] == "Bearer"
    body = response.json()
    assert body.keys() >= PROBLEM_KEYS
    assert body["type"] == "urn:ledgerflow:problem:unauthorized"
    assert body["status"] == 401
    assert body["instance"] == "/api/v1/me"
    assert body["request_id"] == response.headers["x-request-id"]


async def test_unknown_route_is_problem_json(client: AsyncClient) -> None:
    response = await client.get("/api/v1/nope")

    assert response.status_code == 404
    assert response.json()["type"] == "urn:ledgerflow:problem:not-found"


async def test_validation_errors_are_listed(client: AsyncClient) -> None:
    response = await client.post("/api/v1/auth/token", json={"email": "a@b.c"})

    assert response.status_code == 422
    body = response.json()
    assert body["type"] == "urn:ledgerflow:problem:validation-error"
    assert body["errors"][0]["loc"] == ["body", "password"]


async def test_internal_endpoints_require_service_token(client: AsyncClient) -> None:
    response = await client.post(
        "/api/v1/internal/documents/00000000-0000-0000-0000-000000000000/status",
        json={"status": "processing"},
        headers={"Authorization": "Bearer wrong"},
    )
    assert response.status_code == 401


async def test_caller_request_id_is_echoed_only_when_safe(client: AsyncClient) -> None:
    echoed = await client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert echoed.headers["x-request-id"] == "abc-123"

    replaced = await client.get("/healthz", headers={"X-Request-ID": "bad id\r\nx"})
    assert replaced.headers["x-request-id"] != "bad id\r\nx"
    assert len(replaced.headers["x-request-id"]) == 32
