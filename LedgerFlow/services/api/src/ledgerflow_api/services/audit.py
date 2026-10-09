"""Append-only audit trail. Callers write events in the same transaction as the change itself,
so the trail can never disagree with the data it describes."""

import uuid
from collections.abc import Mapping

from sqlalchemy.ext.asyncio import AsyncSession

from ledgerflow_api.context import request_id_var
from ledgerflow_api.db.models import AuditEvent
from ledgerflow_api.domain.fields import JsonValue

type Diff = dict[str, dict[str, JsonValue]]


def diff_snapshots(before: Mapping[str, JsonValue], after: Mapping[str, JsonValue]) -> Diff | None:
    """Field-level ``{key: {before, after}}`` for changed keys, or ``None`` if nothing changed."""
    changes: Diff = {}
    for key in sorted(before.keys() | after.keys()):
        old, new = before.get(key), after.get(key)
        if old != new:
            changes[key] = {"before": old, "after": new}
    return changes or None


def record_audit(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    actor_id: uuid.UUID | None,
    action: str,
    entity_id: uuid.UUID,
    diff: Diff | None = None,
    entity_type: str = "document",
) -> AuditEvent:
    event = AuditEvent(
        tenant_id=tenant_id,
        actor_id=actor_id,
        action=action,
        entity_type=entity_type,
        entity_id=entity_id,
        diff=diff,
        request_id=request_id_var.get(),
    )
    session.add(event)
    return event
