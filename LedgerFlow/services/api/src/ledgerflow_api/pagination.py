"""Opaque keyset cursors over ``(created_at, id)``.

Offsets drift when rows are inserted between page requests and get slower with depth; a keyset
on the composite index ``(tenant_id, created_at, id)`` does neither. The cursor also records the
sort direction so a cursor can't be replayed against a differently ordered query.
"""

import base64
import binascii
import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

SortOrder = Literal["-created_at", "created_at"]


class InvalidCursorError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Cursor:
    created_at: datetime
    id: uuid.UUID
    sort: SortOrder


def encode_cursor(cursor: Cursor) -> str:
    payload = {"c": cursor.created_at.isoformat(), "i": str(cursor.id), "s": cursor.sort}
    raw = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def decode_cursor(token: str, *, expected_sort: SortOrder) -> Cursor:
    try:
        padded = token + "=" * (-len(token) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")))
        created_at = datetime.fromisoformat(payload["c"])
        cursor = Cursor(created_at=created_at, id=uuid.UUID(payload["i"]), sort=payload["s"])
    except (binascii.Error, UnicodeError, ValueError, KeyError, TypeError) as exc:
        raise InvalidCursorError("cursor is malformed") from exc
    if created_at.tzinfo is None:
        raise InvalidCursorError("cursor timestamp must be timezone-aware")
    if cursor.sort != expected_sort:
        raise InvalidCursorError("cursor was issued for a different sort order")
    return cursor
