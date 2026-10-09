"""Transactional outbox writer.

Events are rows inserted in the caller's transaction; ``workers.outbox_relay`` publishes them
afterwards. Publishing directly from a request would either lose the event when the broker is
down after commit, or announce a document that a later rollback erased.
"""

import uuid
from datetime import UTC, datetime

from opentelemetry.propagate import inject
from sqlalchemy.ext.asyncio import AsyncSession

from ledgerflow_api.db.models import Document, OutboxEvent

DOCUMENT_UPLOADED = "document.uploaded"


def utc_isoformat(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _current_traceparent() -> str | None:
    carrier: dict[str, str] = {}
    inject(carrier)
    return carrier.get("traceparent")


def enqueue_document_uploaded(session: AsyncSession, document: Document) -> OutboxEvent:
    # The outbox row id doubles as the event id, so the relay can use it as the AMQP message id
    # and consumers can deduplicate on it.
    event_id = uuid.uuid4()
    event = OutboxEvent(
        id=event_id,
        tenant_id=document.tenant_id,
        event_type=DOCUMENT_UPLOADED,
        payload={
            "event_id": str(event_id),
            "event_type": DOCUMENT_UPLOADED,
            "occurred_at": utc_isoformat(datetime.now(UTC)),
            "tenant_id": str(document.tenant_id),
            "document_id": str(document.id),
            "storage_key": document.storage_key,
            "content_type": document.content_type,
            "sha256": document.sha256,
            # Broker-level retries increment this; every event the API emits starts a fresh run.
            "attempt": 1,
            "trace_parent": _current_traceparent(),
        },
    )
    session.add(event)
    return event
