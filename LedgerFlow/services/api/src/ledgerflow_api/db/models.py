"""ORM models. The schema itself (including RLS policies and grants) lives in migrations/."""

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Double,
    Enum,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ledgerflow_api.db.base import Base, created_at_column, uuid_pk
from ledgerflow_api.domain.lifecycle import DocumentStatus
from ledgerflow_api.domain.roles import Role

__all__ = [
    "AuditEvent",
    "Base",
    "Document",
    "Extraction",
    "OutboxEvent",
    "ProcessedEvent",
    "Tenant",
    "User",
]


def _pg_enum(enum: type[StrEnum], name: str) -> Enum:
    # Persist enum *values* ("needs_review"), not member names ("NEEDS_REVIEW").
    return Enum(enum, name=name, values_callable=lambda members: [m.value for m in members])


document_status_enum = _pg_enum(DocumentStatus, "document_status")
user_role_enum = _pg_enum(Role, "user_role")
doc_type_enum = Enum("invoice", "nfe", name="doc_type")


class Tenant(Base):
    __tablename__ = "tenants"
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    id: Mapped[uuid.UUID] = uuid_pk()
    name: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    auto_approve_threshold: Mapped[Decimal] = mapped_column(
        Numeric(3, 2), server_default=text("0.85")
    )
    created_at: Mapped[datetime] = created_at_column()

    __table_args__ = (
        CheckConstraint(
            "auto_approve_threshold BETWEEN 0 AND 1", name="auto_approve_threshold_range"
        ),
    )


class User(Base):
    __tablename__ = "users"
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    id: Mapped[uuid.UUID] = uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    email: Mapped[str] = mapped_column(String(320))
    name: Mapped[str] = mapped_column(String(200))
    role: Mapped[Role] = mapped_column(user_role_enum)
    password_hash: Mapped[str] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(server_default=text("true"))
    created_at: Mapped[datetime] = created_at_column()

    __table_args__ = (
        # Login is by email alone, so addresses are unique across tenants.
        Index("uq_users_email_lower", func.lower(email), unique=True),
        Index("ix_users_tenant_id_created_at", "tenant_id", "created_at"),
    )


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    uploaded_by_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(Text)
    doc_type: Mapped[str | None] = mapped_column(doc_type_enum)
    status: Mapped[DocumentStatus] = mapped_column(document_status_enum)

    # Points at the current Extraction.version; bumping it is what makes a correction change the
    # document row (and therefore its version/ETag).
    extraction_version: Mapped[int | None] = mapped_column(Integer)
    # Denormalized from the current extraction so list queries never touch JSONB.
    supplier_name: Mapped[str | None] = mapped_column(String(255))
    total_amount: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))
    currency: Mapped[str | None] = mapped_column(String(3))
    overall_confidence: Mapped[float | None] = mapped_column(Double)
    # Decided once, when the worker's extraction lands; later threshold changes don't rewrite
    # history.
    review_required: Mapped[bool | None]

    rejection_reason: Mapped[str | None] = mapped_column(Text)
    last_error: Mapped[str | None] = mapped_column(Text)
    reviewed_by_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    reviewed_at: Mapped[datetime | None]
    processed_at: Mapped[datetime | None]

    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = created_at_column()
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())

    uploaded_by: Mapped[User] = relationship(foreign_keys=[uploaded_by_id], lazy="raise")

    # Every UPDATE is issued as "... WHERE version = :old" and bumps the counter, which backs
    # the ETag/If-Match contract even for writers that forget to lock the row.
    __mapper_args__ = {  # noqa: RUF012
        "version_id_col": version,
        "eager_defaults": True,
    }
    __table_args__ = (
        UniqueConstraint("tenant_id", "sha256", name="uq_documents_tenant_id_sha256"),
        Index("ix_documents_tenant_id_created_at_id", "tenant_id", "created_at", "id"),
        Index("ix_documents_tenant_id_status", "tenant_id", "status"),
    )


class Extraction(Base):
    """Append-only: corrections create a new version, nothing is updated in place."""

    __tablename__ = "extractions"
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    id: Mapped[uuid.UUID] = uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    version: Mapped[int] = mapped_column(Integer)
    model: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(100))
    fields: Mapped[dict[str, Any]]
    overall_confidence: Mapped[float] = mapped_column(Double)
    cost_usd_micros: Mapped[int] = mapped_column(BigInteger)
    latency_ms: Mapped[int] = mapped_column(Integer)
    created_by_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    created_at: Mapped[datetime] = created_at_column()

    __table_args__ = (
        UniqueConstraint("document_id", "version", name="uq_extractions_document_id_version"),
        Index("ix_extractions_tenant_id_document_id", "tenant_id", "document_id"),
        Index("ix_extractions_tenant_id_created_at", "tenant_id", "created_at"),
    )


class AuditEvent(Base):
    """Append-only; the application role is granted INSERT and SELECT only."""

    __tablename__ = "audit_events"
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    id: Mapped[uuid.UUID] = uuid_pk()
    # Events written in one transaction share created_at (now() is the transaction start), and
    # random UUIDs don't sort, so an identity column provides a stable write order.
    seq: Mapped[int] = mapped_column(BigInteger, Identity(always=True))
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    actor_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    action: Mapped[str] = mapped_column(String(100))
    entity_type: Mapped[str] = mapped_column(String(50))
    entity_id: Mapped[uuid.UUID]
    diff: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True))
    request_id: Mapped[str | None] = mapped_column(String(128))
    created_at: Mapped[datetime] = created_at_column()

    actor: Mapped[User | None] = relationship(lazy="raise")

    __table_args__ = (
        Index(
            "ix_audit_events_tenant_id_entity",
            "tenant_id",
            "entity_type",
            "entity_id",
            "seq",
        ),
    )


class OutboxEvent(Base):
    __tablename__ = "outbox"
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    id: Mapped[uuid.UUID] = uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(100))
    payload: Mapped[dict[str, Any]]
    attempts: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = created_at_column()
    published_at: Mapped[datetime | None]

    __table_args__ = (
        Index("ix_outbox_tenant_id_created_at", "tenant_id", "created_at"),
        # The relay only ever scans unpublished rows; a partial index keeps that scan tiny.
        Index(
            "ix_outbox_unpublished",
            "created_at",
            postgresql_where=text("published_at IS NULL"),
        ),
    )


class ProcessedEvent(Base):
    __tablename__ = "processed_events"
    __mapper_args__ = {"eager_defaults": True}  # noqa: RUF012

    event_id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tenants.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(100))
    processed_at: Mapped[datetime] = created_at_column()

    __table_args__ = (
        Index("ix_processed_events_tenant_id_processed_at", "tenant_id", "processed_at"),
    )
