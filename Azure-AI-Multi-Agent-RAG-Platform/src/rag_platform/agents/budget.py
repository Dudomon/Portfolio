"""Per-request token budget with reservations.

A call reserves its worst case (estimated prompt + `max_output_tokens`) before it is
made, and the reservation is replaced by the actual usage once the response arrives.
Reserving up front is what makes the budget hold under concurrency: the retriever
issues several calls in parallel, and checking only after the fact would let them
all pass the check and jointly overshoot.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from rag_platform.domain import TokenUsage
from rag_platform.telemetry.cost import CostLedger


class BudgetExhaustedError(Exception):
    pass


class Reservation:
    def __init__(self, budget: TokenBudget, tokens: int) -> None:
        self._budget = budget
        self.tokens = tokens
        self.settled = False

    def settle(self, operation: str, deployment: str, usage: TokenUsage) -> None:
        self._budget.settle(self, operation, deployment, usage)


class TokenBudget:
    def __init__(self, limit: int, ledger: CostLedger) -> None:
        self.limit = limit
        self.ledger = ledger
        self._reserved = 0

    @property
    def used(self) -> int:
        return self.ledger.total_usage.total

    @property
    def remaining(self) -> int:
        return self.limit - self.used - self._reserved

    @contextmanager
    def reserve(self, tokens: int) -> Iterator[Reservation]:
        if tokens > self.remaining:
            raise BudgetExhaustedError(
                f"Call needs up to {tokens} tokens but only {self.remaining} of {self.limit} remain"
            )
        reservation = Reservation(self, tokens)
        self._reserved += tokens
        try:
            yield reservation
        finally:
            # Unsettled reservations (errors, cancellation) are released. Tokens a
            # canceled call already consumed are not reported by the API, so the
            # ledger can under-count in that case; the budget still bounds spend.
            if not reservation.settled:
                self._reserved -= tokens

    def settle(
        self, reservation: Reservation, operation: str, deployment: str, usage: TokenUsage
    ) -> None:
        if reservation.settled:
            raise RuntimeError("Reservation already settled")
        reservation.settled = True
        self._reserved -= reservation.tokens
        self.ledger.record(operation, deployment, usage)
