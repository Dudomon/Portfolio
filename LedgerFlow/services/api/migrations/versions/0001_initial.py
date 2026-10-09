"""Initial schema: tenants, users, documents, extractions, audit, outbox, idempotency.

Also creates the unprivileged ``ledgerflow_app`` role used by tenant-scoped sessions, grants it
the minimum it needs (append-only tables get no UPDATE/DELETE) and enables Row-Level Security
on every tenant-owned table.

Revision ID: 0001
Revises:
Create Date: 2026-10-09
"""

import uuid
from collections.abc import Sequence
from datetime import datetime

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

APP_ROLE = "ledgerflow_app"
TENANT_SETTING = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"

document_status = postgresql.ENUM(
    "uploaded",
    "queued",
    "processing",
    "extracted",
    "needs_review",
    "approved",
    "rejected",
    "exported",
    "failed",
    name="document_status",
    create_type=False,
)
user_role = postgresql.ENUM("viewer", "reviewer", "admin", name="user_role", create_type=False)
doc_type = postgresql.ENUM("invoice", "nfe", name="doc_type", create_type=False)

TENANT_OWNED_TABLES = (
    "users",
    "documents",
    "extractions",
    "audit_events",
    "outbox",
    "processed_events",
)


def _created_at(name: str = "created_at") -> sa.Column[datetime]:
    return sa.Column(name, sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False)


def _tenant_fk(table: str) -> sa.Column[uuid.UUID]:
    return sa.Column(
        "tenant_id",
        sa.Uuid(),
        sa.ForeignKey("tenants.id", ondelete="CASCADE", name=f"fk_{table}_tenant_id_tenants"),
        nullable=False,
    )


def upgrade() -> None:
    bind = op.get_bind()
    for enum in (document_status, user_role, doc_type):
        enum.create(bind)

    op.create_table(
        "tenants",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("slug", sa.String(64), nullable=False),
        sa.Column(
            "auto_approve_threshold",
            sa.Numeric(3, 2),
            server_default=sa.text("0.85"),
            nullable=False,
        ),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_tenants"),
        sa.UniqueConstraint("slug", name="uq_tenants_slug"),
        sa.CheckConstraint(
            "auto_approve_threshold BETWEEN 0 AND 1",
            name="ck_tenants_auto_approve_threshold_range",
        ),
    )

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        _tenant_fk("users"),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("role", user_role, nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=False),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true"), nullable=False),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
    )
    op.create_index("uq_users_email_lower", "users", [sa.text("lower(email)")], unique=True)
    op.create_index("ix_users_tenant_id_created_at", "users", ["tenant_id", "created_at"])

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        _tenant_fk("documents"),
        sa.Column(
            "uploaded_by_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", name="fk_documents_uploaded_by_id_users"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("storage_key", sa.Text(), nullable=False),
        sa.Column("doc_type", doc_type, nullable=True),
        sa.Column("status", document_status, nullable=False),
        sa.Column("extraction_version", sa.Integer(), nullable=True),
        sa.Column("supplier_name", sa.String(255), nullable=True),
        sa.Column("total_amount", sa.Numeric(18, 2), nullable=True),
        sa.Column("currency", sa.String(3), nullable=True),
        sa.Column("overall_confidence", sa.Double(), nullable=True),
        sa.Column("review_required", sa.Boolean(), nullable=True),
        sa.Column("rejection_reason", sa.Text(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "reviewed_by_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", name="fk_documents_reviewed_by_id_users"),
            nullable=True,
        ),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        _created_at(),
        _created_at("updated_at"),
        sa.PrimaryKeyConstraint("id", name="pk_documents"),
        sa.UniqueConstraint("tenant_id", "sha256", name="uq_documents_tenant_id_sha256"),
    )
    op.create_index(
        "ix_documents_tenant_id_created_at_id", "documents", ["tenant_id", "created_at", "id"]
    )
    op.create_index("ix_documents_tenant_id_status", "documents", ["tenant_id", "status"])

    op.create_table(
        "extractions",
        sa.Column("id", sa.Uuid(), nullable=False),
        _tenant_fk("extractions"),
        sa.Column(
            "document_id",
            sa.Uuid(),
            sa.ForeignKey(
                "documents.id", ondelete="CASCADE", name="fk_extractions_document_id_documents"
            ),
            nullable=False,
        ),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("prompt_version", sa.String(100), nullable=True),
        sa.Column("fields", postgresql.JSONB(), nullable=False),
        sa.Column("overall_confidence", sa.Double(), nullable=False),
        sa.Column("cost_usd_micros", sa.BigInteger(), nullable=False),
        sa.Column("latency_ms", sa.Integer(), nullable=False),
        sa.Column(
            "created_by_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", name="fk_extractions_created_by_id_users"),
            nullable=True,
        ),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_extractions"),
        sa.UniqueConstraint("document_id", "version", name="uq_extractions_document_id_version"),
    )
    op.create_index(
        "ix_extractions_tenant_id_document_id", "extractions", ["tenant_id", "document_id"]
    )
    op.create_index(
        "ix_extractions_tenant_id_created_at", "extractions", ["tenant_id", "created_at"]
    )

    op.create_table(
        "audit_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("seq", sa.BigInteger(), sa.Identity(always=True), nullable=False),
        _tenant_fk("audit_events"),
        sa.Column(
            "actor_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", name="fk_audit_events_actor_id_users"),
            nullable=True,
        ),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", sa.Uuid(), nullable=False),
        sa.Column("diff", postgresql.JSONB(), nullable=True),
        sa.Column("request_id", sa.String(128), nullable=True),
        _created_at(),
        sa.PrimaryKeyConstraint("id", name="pk_audit_events"),
    )
    op.create_index(
        "ix_audit_events_tenant_id_entity",
        "audit_events",
        ["tenant_id", "entity_type", "entity_id", "seq"],
    )

    op.create_table(
        "outbox",
        sa.Column("id", sa.Uuid(), nullable=False),
        _tenant_fk("outbox"),
        sa.Column("event_type", sa.String(100), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("attempts", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("last_error", sa.Text(), nullable=True),
        _created_at(),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_outbox"),
    )
    op.create_index("ix_outbox_tenant_id_created_at", "outbox", ["tenant_id", "created_at"])
    op.create_index(
        "ix_outbox_unpublished",
        "outbox",
        ["created_at"],
        postgresql_where=sa.text("published_at IS NULL"),
    )

    op.create_table(
        "processed_events",
        sa.Column("event_id", sa.Uuid(), nullable=False),
        _tenant_fk("processed_events"),
        sa.Column("event_type", sa.String(100), nullable=False),
        _created_at("processed_at"),
        sa.PrimaryKeyConstraint("event_id", name="pk_processed_events"),
    )
    op.create_index(
        "ix_processed_events_tenant_id_processed_at",
        "processed_events",
        ["tenant_id", "processed_at"],
    )

    _create_app_role()
    _enable_row_level_security()


def _create_app_role() -> None:
    # Roles are cluster-wide, so tolerate one left behind by another database or a downgrade.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                CREATE ROLE {APP_ROLE} NOLOGIN;
            END IF;
        END
        $$
        """
    )
    # The connecting (owner) role switches into the app role per transaction via SET LOCAL ROLE.
    op.execute(f"GRANT {APP_ROLE} TO CURRENT_USER")
    op.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
    op.execute(f"GRANT SELECT ON tenants, users TO {APP_ROLE}")
    op.execute(f"GRANT SELECT, INSERT, UPDATE ON documents TO {APP_ROLE}")
    # Append-only by privilege, not just by convention.
    op.execute(f"GRANT SELECT, INSERT ON extractions, audit_events, outbox TO {APP_ROLE}")


def _enable_row_level_security() -> None:
    op.execute("ALTER TABLE tenants ENABLE ROW LEVEL SECURITY")
    op.execute(
        f"CREATE POLICY tenant_isolation ON tenants "
        f"USING (id = {TENANT_SETTING}) WITH CHECK (id = {TENANT_SETTING})"
    )
    for table in TENANT_OWNED_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING (tenant_id = {TENANT_SETTING}) WITH CHECK (tenant_id = {TENANT_SETTING})"
        )


def downgrade() -> None:
    for table in (
        "processed_events",
        "outbox",
        "audit_events",
        "extractions",
        "documents",
        "users",
    ):
        op.drop_table(table)
    op.drop_table("tenants")
    bind = op.get_bind()
    for enum in (doc_type, user_role, document_status):
        enum.drop(bind)
    # The role is cluster-wide and may serve other databases, so it is intentionally kept.
    op.execute(f"REVOKE USAGE ON SCHEMA public FROM {APP_ROLE}")
