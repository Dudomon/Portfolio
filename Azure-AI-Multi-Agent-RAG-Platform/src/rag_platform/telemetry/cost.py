"""Per-request token and cost accounting."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from rag_platform.config import ModelPrice
from rag_platform.domain import TokenUsage


class UsageEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    operation: str
    deployment: str
    usage: TokenUsage
    cost_usd: float


class CostLedger:
    """Collects the usage of every model call made while serving one request.

    Unknown deployments are priced at zero and reported via `unpriced_deployments`
    rather than failing the request; cost reporting must never take the service down.
    """

    def __init__(self, prices: Mapping[str, ModelPrice]) -> None:
        self._prices = prices
        self._entries: list[UsageEntry] = []
        self.unpriced_deployments: set[str] = set()

    def record(self, operation: str, deployment: str, usage: TokenUsage) -> UsageEntry:
        price = self._prices.get(deployment)
        if price is None:
            self.unpriced_deployments.add(deployment)
            cost = 0.0
        else:
            cost = (
                usage.input_tokens * price.input_per_million
                + usage.output_tokens * price.output_per_million
            ) / 1_000_000
        entry = UsageEntry(operation=operation, deployment=deployment, usage=usage, cost_usd=cost)
        self._entries.append(entry)
        return entry

    @property
    def entries(self) -> tuple[UsageEntry, ...]:
        return tuple(self._entries)

    @property
    def total_usage(self) -> TokenUsage:
        return sum((e.usage for e in self._entries), TokenUsage())

    @property
    def total_cost_usd(self) -> float:
        return round(sum(e.cost_usd for e in self._entries), 6)
