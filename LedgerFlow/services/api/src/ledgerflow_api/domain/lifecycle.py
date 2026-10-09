"""Document lifecycle as an explicit state machine.

The transition table is the single authority on which status changes are legal; services
never assign ``Document.status`` directly.
"""

from enum import StrEnum
from types import MappingProxyType


class DocumentStatus(StrEnum):
    UPLOADED = "uploaded"
    QUEUED = "queued"
    PROCESSING = "processing"
    EXTRACTED = "extracted"
    NEEDS_REVIEW = "needs_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPORTED = "exported"
    FAILED = "failed"


_S = DocumentStatus

ALLOWED_TRANSITIONS: MappingProxyType[DocumentStatus, frozenset[DocumentStatus]] = MappingProxyType(
    {
        _S.UPLOADED: frozenset({_S.QUEUED}),
        _S.QUEUED: frozenset({_S.PROCESSING}),
        _S.PROCESSING: frozenset({_S.EXTRACTED, _S.NEEDS_REVIEW, _S.FAILED}),
        _S.EXTRACTED: frozenset({_S.APPROVED}),
        _S.NEEDS_REVIEW: frozenset({_S.APPROVED, _S.REJECTED}),
        _S.APPROVED: frozenset({_S.EXPORTED}),
        _S.REJECTED: frozenset(),
        _S.EXPORTED: frozenset(),
        _S.FAILED: frozenset({_S.QUEUED}),
    }
)

# Statuses in which a reviewer may still change extracted values.
CORRECTABLE_STATUSES = frozenset({_S.EXTRACTED, _S.NEEDS_REVIEW})

# Statuses the extraction worker is allowed to report through the internal status endpoint;
# everything else is decided by the API itself.
WORKER_REPORTABLE_STATUSES = frozenset({_S.PROCESSING, _S.FAILED, _S.EXPORTED})


class InvalidTransitionError(Exception):
    def __init__(self, current: DocumentStatus, target: DocumentStatus) -> None:
        super().__init__(f"cannot move document from '{current}' to '{target}'")
        self.current = current
        self.target = target


def can_transition(current: DocumentStatus, target: DocumentStatus) -> bool:
    return target in ALLOWED_TRANSITIONS[current]


def ensure_transition(current: DocumentStatus, target: DocumentStatus) -> DocumentStatus:
    """Return ``target`` if the move is legal, otherwise raise ``InvalidTransitionError``."""
    if not can_transition(current, target):
        raise InvalidTransitionError(current, target)
    return target


def review_outcome(
    *, overall_confidence: float, threshold: float, has_issues: bool
) -> DocumentStatus:
    """Decide where a fresh extraction lands.

    ``overall_confidence`` is the minimum field confidence, so comparing it to the threshold is
    equivalent to "every field clears the threshold". Rule violations (bad check digits,
    inconsistent totals) force human review regardless of how confident the model was.
    """
    if has_issues or overall_confidence < threshold:
        return _S.NEEDS_REVIEW
    return _S.EXTRACTED
