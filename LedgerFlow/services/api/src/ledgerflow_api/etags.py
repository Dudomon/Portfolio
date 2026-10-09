"""ETag rendering and If-Match evaluation for optimistic concurrency.

The contract fixes weak ETags (``W/"<version>"``). RFC 9110 prescribes strong comparison for
If-Match, under which a weak tag never matches; since our validator is a semantic row version
rather than a byte-level hash, we compare opaque tags and ignore the weakness flag.
"""

from ledgerflow_api.errors import PreconditionFailedError, PreconditionRequiredError


def etag_for(version: int) -> str:
    return f'W/"{version}"'


def _opaque_tags(header: str) -> list[str]:
    tags = []
    for part in header.split(","):
        tag = part.strip().removeprefix("W/")
        if len(tag) >= 2 and tag.startswith('"') and tag.endswith('"'):
            tags.append(tag[1:-1])
    return tags


def check_if_match(if_match: str | None, current_version: int) -> None:
    """Raise 428 if the header is missing, 412 if no listed tag matches ``current_version``."""
    if if_match is None or not if_match.strip():
        raise PreconditionRequiredError("This operation requires an If-Match header.")
    if if_match.strip() == "*":
        return
    if str(current_version) not in _opaque_tags(if_match):
        raise PreconditionFailedError(
            "The document was modified since it was read; reload it and retry.",
            extensions={"current_etag": etag_for(current_version)},
        )
