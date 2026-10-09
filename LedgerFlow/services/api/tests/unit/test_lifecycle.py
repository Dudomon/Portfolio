from collections import deque

import pytest

from ledgerflow_api.domain.lifecycle import (
    ALLOWED_TRANSITIONS,
    DocumentStatus,
    InvalidTransitionError,
    can_transition,
    ensure_transition,
    review_outcome,
)

S = DocumentStatus

EXPECTED_EDGES = {
    (S.UPLOADED, S.QUEUED),
    (S.QUEUED, S.PROCESSING),
    (S.PROCESSING, S.EXTRACTED),
    (S.PROCESSING, S.NEEDS_REVIEW),
    (S.PROCESSING, S.FAILED),
    (S.EXTRACTED, S.APPROVED),
    (S.NEEDS_REVIEW, S.APPROVED),
    (S.NEEDS_REVIEW, S.REJECTED),
    (S.APPROVED, S.EXPORTED),
    (S.FAILED, S.QUEUED),
}


def test_table_covers_every_status() -> None:
    assert set(ALLOWED_TRANSITIONS) == set(DocumentStatus)


def test_table_matches_documented_lifecycle() -> None:
    edges = {(src, dst) for src, targets in ALLOWED_TRANSITIONS.items() for dst in targets}
    assert edges == EXPECTED_EDGES


@pytest.mark.parametrize("current", list(DocumentStatus))
@pytest.mark.parametrize("target", list(DocumentStatus))
def test_ensure_transition_agrees_with_table(current: S, target: S) -> None:
    if (current, target) in EXPECTED_EDGES:
        assert ensure_transition(current, target) is target
    else:
        with pytest.raises(InvalidTransitionError) as excinfo:
            ensure_transition(current, target)
        assert (excinfo.value.current, excinfo.value.target) == (current, target)


@pytest.mark.parametrize("terminal", [S.REJECTED, S.EXPORTED])
def test_terminal_statuses_have_no_exits(terminal: S) -> None:
    assert not any(can_transition(terminal, target) for target in DocumentStatus)


def test_every_status_is_reachable_from_upload() -> None:
    seen, frontier = {S.UPLOADED}, deque([S.UPLOADED])
    while frontier:
        for nxt in ALLOWED_TRANSITIONS[frontier.popleft()]:
            if nxt not in seen:
                seen.add(nxt)
                frontier.append(nxt)
    assert seen == set(DocumentStatus)


@pytest.mark.parametrize(
    ("confidence", "has_issues", "expected"),
    [
        (0.85, False, S.EXTRACTED),  # the threshold itself is inclusive
        (0.99, False, S.EXTRACTED),
        (0.8499, False, S.NEEDS_REVIEW),
        (0.99, True, S.NEEDS_REVIEW),  # rule violations override confidence
        (0.0, False, S.NEEDS_REVIEW),
    ],
)
def test_review_outcome(confidence: float, has_issues: bool, expected: S) -> None:
    assert (
        review_outcome(overall_confidence=confidence, threshold=0.85, has_issues=has_issues)
        is expected
    )
