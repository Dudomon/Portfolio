"""Engine and session factories, including tenant scoping for Row-Level Security.

A tenant-scoped session switches to the unprivileged ``ledgerflow_app`` role and publishes the
tenant id via ``SET LOCAL`` at the start of *every* transaction it opens. Doing this in an
``after_begin`` hook (instead of once per request) matters: ``SET LOCAL`` dies with the
transaction, so a request that commits and keeps querying would otherwise continue as the
table owner, which bypasses RLS.

Sessions without a tenant run as the connection's own role (the schema owner) and are reserved
for work that is cross-tenant by nature: login, the outbox relay and the internal endpoints.
"""

import uuid
from typing import Any

from sqlalchemy import event, text
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session, SessionTransaction

from ledgerflow_api.config import Settings

APP_DB_ROLE = "ledgerflow_app"
TENANT_ID_KEY = "tenant_id"

_SET_ROLE = text(f"SET LOCAL ROLE {APP_DB_ROLE}")
# set_config(..., is_local => true) is SET LOCAL with a bindable value.
_SET_TENANT = text("SELECT set_config('app.tenant_id', :tenant_id, true)")


class ScopedSession(Session):
    """Sync session class behind every ``AsyncSession``; carries the RLS hook below."""


@event.listens_for(ScopedSession, "after_begin")
def _apply_tenant_scope(
    session: Session, _transaction: SessionTransaction, connection: Connection
) -> None:
    tenant_id: uuid.UUID | None = session.info.get(TENANT_ID_KEY)
    if tenant_id is None:
        return
    connection.execute(_SET_ROLE)
    connection.execute(_SET_TENANT, {"tenant_id": str(tenant_id)})


def create_engine(settings: Settings) -> AsyncEngine:
    return create_async_engine(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_pre_ping=True,
    )


type SessionFactory = async_sessionmaker[AsyncSession]


def create_session_factory(engine: AsyncEngine) -> SessionFactory:
    return async_sessionmaker(
        engine,
        sync_session_class=ScopedSession,
        expire_on_commit=False,
        autoflush=False,
    )


def tenant_session(factory: SessionFactory, tenant_id: uuid.UUID) -> AsyncSession:
    info: dict[str, Any] = {TENANT_ID_KEY: tenant_id}
    return factory(info=info)


def system_session(factory: SessionFactory) -> AsyncSession:
    return factory()
