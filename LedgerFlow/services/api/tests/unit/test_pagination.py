import base64
import json
import uuid
from datetime import UTC, datetime

import pytest

from ledgerflow_api.pagination import Cursor, InvalidCursorError, decode_cursor, encode_cursor

CURSOR = Cursor(
    created_at=datetime(2026, 10, 9, 12, 0, 0, 123456, tzinfo=UTC),
    id=uuid.UUID("0b6f8f1e-6c3e-4f43-9d2a-6b1f4a3c2d10"),
    sort="-created_at",
)


def _raw(payload: object) -> str:
    return base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")


def test_round_trip_preserves_microseconds() -> None:
    token = encode_cursor(CURSOR)
    assert "=" not in token
    assert decode_cursor(token, expected_sort="-created_at") == CURSOR


def test_cursor_is_bound_to_sort_order() -> None:
    with pytest.raises(InvalidCursorError, match="sort"):
        decode_cursor(encode_cursor(CURSOR), expected_sort="created_at")


@pytest.mark.parametrize(
    "token",
    [
        "not base64 at all!",
        _raw(["a", "list"]),
        _raw({"c": "2026-10-09T12:00:00+00:00", "s": "-created_at"}),
        _raw({"c": "yesterday", "i": str(CURSOR.id), "s": "-created_at"}),
        _raw({"c": "2026-10-09T12:00:00+00:00", "i": "not-a-uuid", "s": "-created_at"}),
        _raw({"c": "2026-10-09T12:00:00", "i": str(CURSOR.id), "s": "-created_at"}),
    ],
)
def test_malformed_cursors_are_rejected(token: str) -> None:
    with pytest.raises(InvalidCursorError):
        decode_cursor(token, expected_sort="-created_at")
