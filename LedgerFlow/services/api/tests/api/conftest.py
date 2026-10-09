"""Fixtures for API tests against a real PostgreSQL (RLS, enums and SKIP LOCKED need it).

The database comes from testcontainers by default. Setting ``TEST_DATABASE_URL`` points the
suite at an existing, dedicated database instead (e.g. a CI service container); it is migrated
and then truncated between tests, so never point it at data you care about.
"""

import os
import uuid
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from testcontainers.community.postgres import PostgresContainer

from ledgerflow_api.api.deps import AppResources, get_storage
from ledgerflow_api.config import Settings
from ledgerflow_api.db.models import Base, Tenant, User
from ledgerflow_api.db.session import system_session
from ledgerflow_api.domain.roles import Role
from ledgerflow_api.main import create_app
from ledgerflow_api.security import hash_password
from tests.conftest import make_settings

API_ROOT = Path(__file__).resolve().parents[2]
PASSWORD = "correct-horse-battery"
_PASSWORD_HASH = hash_password(PASSWORD)


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    if url := os.environ.get("TEST_DATABASE_URL"):
        yield url
        return
    try:
        container = PostgresContainer("postgres:16-alpine", driver="asyncpg")
        container.start()
    except Exception as exc:  # any Docker failure means the suite can't run here
        pytest.skip(f"PostgreSQL container unavailable ({exc}); set TEST_DATABASE_URL instead")
    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture(scope="session")
def migrated_database_url(database_url: str) -> str:
    # Running the real migration (not metadata.create_all) means RLS policies and grants are
    # exercised by every API test.
    config = Config(str(API_ROOT / "alembic.ini"))
    config.attributes["database_url"] = database_url
    command.upgrade(config, "head")
    return database_url


class InMemoryStorage:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

    async def put(self, key: str, data: bytes, *, content_type: str) -> None:
        self.objects[key] = (data, content_type)

    async def presigned_get_url(self, key: str, *, filename: str, content_type: str) -> str:
        return f"https://storage.test/{key}?filename={filename}"


@dataclass(frozen=True)
class Account:
    id: uuid.UUID
    email: str
    role: Role


@dataclass(frozen=True)
class SeededTenant:
    id: uuid.UUID
    accounts: dict[Role, Account] = field(default_factory=dict)


@pytest.fixture
def settings(migrated_database_url: str) -> Settings:
    return make_settings(database_url=migrated_database_url, max_upload_bytes=64 * 1024)


@pytest.fixture
def storage() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.fixture
async def app(settings: Settings, storage: InMemoryStorage) -> AsyncIterator[FastAPI]:
    application = create_app(settings)
    application.dependency_overrides[get_storage] = lambda: storage
    async with LifespanManager(application):
        resources: AppResources = application.state.resources
        tables = ", ".join(t.name for t in Base.metadata.sorted_tables)
        async with resources.engine.begin() as connection:
            await connection.execute(text(f"TRUNCATE {tables} CASCADE"))
        yield application


@pytest.fixture
def resources(app: FastAPI) -> AppResources:
    return app.state.resources  # type: ignore[no-any-return]


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test/api/v1") as http:
        yield http


async def _seed_tenant(resources: AppResources, slug: str) -> SeededTenant:
    async with system_session(resources.session_factory) as session:
        tenant = Tenant(name=slug.title(), slug=slug)
        session.add(tenant)
        await session.flush()
        accounts = {}
        for role in Role:
            user = User(
                tenant_id=tenant.id,
                email=f"{role.value}@{slug}.example.com",
                name=f"{role.value.title()} of {slug}",
                role=role,
                password_hash=_PASSWORD_HASH,
            )
            session.add(user)
            await session.flush()
            accounts[role] = Account(id=user.id, email=user.email, role=role)
        await session.commit()
        return SeededTenant(id=tenant.id, accounts=accounts)


@pytest.fixture
async def tenant(resources: AppResources) -> SeededTenant:
    return await _seed_tenant(resources, "acme")


@pytest.fixture
async def other_tenant(resources: AppResources) -> SeededTenant:
    return await _seed_tenant(resources, "globex")


@dataclass(frozen=True)
class AuthHeaders:
    """Builds bearer headers without paying for an Argon2 login in every test."""

    resources: AppResources

    def __call__(self, seeded: SeededTenant, role: Role, **extra: str) -> dict[str, str]:
        account = seeded.accounts[role]
        issued = self.resources.tokens.issue(user_id=account.id, tenant_id=seeded.id, role=role)
        return {"Authorization": f"Bearer {issued.token}", **extra}


@pytest.fixture
def auth(resources: AppResources) -> AuthHeaders:
    return AuthHeaders(resources)
