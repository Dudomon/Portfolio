"""Audit-trail rows for gate decisions, promotions, and rollbacks.

One append-only Delta table records every governance event with the policy
version and the human or service identity behind it. Rows are normalized here
so the local runner (JSON lines) and Databricks (Delta) write the same shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Literal

from credit_risk.schemas import to_ddl

EventType = Literal["promotion_gate", "registration", "promotion", "rollback"]

AUDIT_COLUMNS: tuple[tuple[str, str], ...] = (
    ("event_time", "TIMESTAMP"),
    ("event_type", "STRING"),
    ("model_name", "STRING"),
    ("model_version", "STRING"),
    ("run_id", "STRING"),
    ("champion_version", "STRING"),
    ("policy_version", "STRING"),
    ("passed", "BOOLEAN"),
    ("failed_checks", "ARRAY<STRING>"),
    ("checks_json", "STRING"),
    ("actor", "STRING"),
)
AUDIT_DDL = to_ddl(AUDIT_COLUMNS)
_COLUMN_NAMES = tuple(name for name, _ in AUDIT_COLUMNS)


def normalize(record: Mapping[str, Any]) -> dict[str, Any]:
    """Order columns, fill absent optional fields with None, reject unknown keys."""
    unknown = set(record) - set(_COLUMN_NAMES)
    if unknown:
        raise ValueError(f"Unknown audit fields: {sorted(unknown)}")
    for required in ("event_time", "event_type", "model_name", "actor"):
        if record.get(required) in (None, ""):
            raise ValueError(f"Audit field '{required}' is required")
    return {name: record.get(name) for name in _COLUMN_NAMES}


def alias_change_record(
    event_type: EventType,
    *,
    model_name: str,
    model_version: str,
    previous_version: str | None,
    actor: str,
) -> dict[str, Any]:
    return normalize(
        {
            "event_time": datetime.now(UTC),
            "event_type": event_type,
            "model_name": model_name,
            "model_version": model_version,
            "champion_version": previous_version,
            "actor": actor,
        }
    )
