"""Declarative data-quality rules for the silver layer.

Each rule produces a boolean *violation* mask. Rows violating any rule are
routed to a quarantine table together with the names of the rules they broke,
so the data owner can fix the source instead of the pipeline silently dropping
records. A batch-level threshold stops the job when too many rows fail, because
training on a mostly-broken batch is worse than not training at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from credit_risk.domain import CROPS, GENDERS, REGIONS
from credit_risk.schemas import DQ_FAILED_RULES_COLUMN


class Rule(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def violations(self, df: pd.DataFrame) -> pd.Series: ...


@dataclass(frozen=True)
class NotNull:
    column: str
    description: str = ""

    @property
    def name(self) -> str:
        return f"{self.column}_not_null"

    def violations(self, df: pd.DataFrame) -> pd.Series:
        return df[self.column].isna()


@dataclass(frozen=True)
class InRange:
    """Inclusive range check. Nulls are violations unless ``allow_null``."""

    column: str
    min_value: float
    max_value: float
    allow_null: bool = False
    exclusive_min: bool = False
    description: str = ""

    @property
    def name(self) -> str:
        return f"{self.column}_in_range"

    def violations(self, df: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(df[self.column], errors="coerce")
        below = values <= self.min_value if self.exclusive_min else values < self.min_value
        out_of_range = (below | (values > self.max_value)).fillna(False).astype(bool)
        if self.allow_null:
            return out_of_range
        return out_of_range | values.isna()


@dataclass(frozen=True)
class InSet:
    column: str
    allowed: frozenset[str | int]
    allow_null: bool = False
    description: str = ""

    @property
    def name(self) -> str:
        return f"{self.column}_in_set"

    def violations(self, df: pd.DataFrame) -> pd.Series:
        series = df[self.column]
        missing = series.isna()
        # Compare on object dtype so Int64/str extension types behave the same.
        outside = ~series.astype(object).isin(list(self.allowed)) & ~missing
        if self.allow_null:
            return outside.astype(bool)
        return (outside | missing).astype(bool)


@dataclass(frozen=True)
class NoRescuedData:
    """Auto Loader puts values that do not match the enforced schema here."""

    column: str = "_rescued_data"
    name: str = "no_rescued_data"
    description: str = "Row contained fields that did not match the bronze schema."

    def violations(self, df: pd.DataFrame) -> pd.Series:
        if self.column not in df.columns:
            return pd.Series(False, index=df.index)
        return df[self.column].notna()


DEFAULT_RULES: tuple[Rule, ...] = (
    NotNull("loan_id", "Primary key of the application."),
    NotNull("producer_id", "Needed for the point-in-time feature lookup."),
    NotNull("application_date", "Needed for time-based splits and lookups."),
    NotNull("has_crop_insurance", "Model feature; no safe default exists."),
    NotNull("irrigated", "Model feature; no safe default exists."),
    NotNull("cooperative_member", "Model feature; no safe default exists."),
    InSet("region", frozenset(REGIONS), description="Macro-region; sensitive attribute."),
    InSet("gender", frozenset(GENDERS), description="Primary holder; sensitive attribute."),
    InSet("crop_type", frozenset(CROPS), description="Financed activity."),
    InRange("farm_size_ha", 0, 100_000, exclusive_min=True),
    InRange("loan_amount_brl", 0, 500_000_000, exclusive_min=True),
    InRange("annual_revenue_brl", 0, 500_000_000, exclusive_min=True),
    InRange("existing_debt_brl", 0, 500_000_000),
    InRange("term_months", 6, 120),
    # Rates are fractions; values above 1 are almost always percent typed in.
    InRange("interest_rate", 0, 0.6),
    InRange("years_farming", 0, 90),
    InRange("credit_history_months", 0, 1_000),
    InRange("prior_defaults", 0, 50),
    InRange("rainfall_anomaly", -6, 6),
    InSet("defaulted", frozenset({0, 1}), allow_null=True, description="Null = not matured."),
    NoRescuedData(),
)


class DataQualityError(RuntimeError):
    """Raised when a batch fails more rules than the configured tolerance."""


@dataclass(frozen=True)
class ValidationResult:
    valid: pd.DataFrame
    quarantined: pd.DataFrame

    @property
    def quarantine_rate(self) -> float:
        total = len(self.valid) + len(self.quarantined)
        return len(self.quarantined) / total if total else 0.0

    def violation_counts(self) -> dict[str, int]:
        exploded = self.quarantined[DQ_FAILED_RULES_COLUMN].explode()
        return {str(k): int(v) for k, v in exploded.value_counts().items()}


def annotate_violations(df: pd.DataFrame, rules: Sequence[Rule] = DEFAULT_RULES) -> pd.DataFrame:
    """Add a list column with the names of the rules each row violates."""
    out = df.copy()
    if out.empty:
        out[DQ_FAILED_RULES_COLUMN] = pd.Series([], dtype=object)
        return out
    names = np.array([rule.name for rule in rules], dtype=object)
    masks = np.column_stack([rule.violations(df).to_numpy(dtype=bool) for rule in rules])
    out[DQ_FAILED_RULES_COLUMN] = [list(names[row]) for row in masks]
    return out


def split_valid(
    annotated: pd.DataFrame, max_quarantine_rate: float | None = None
) -> ValidationResult:
    """Split an annotated frame and enforce the batch-level tolerance."""
    failed = annotated[DQ_FAILED_RULES_COLUMN].map(len) > 0
    result = ValidationResult(
        valid=annotated.loc[~failed].drop(columns=DQ_FAILED_RULES_COLUMN),
        quarantined=annotated.loc[failed],
    )
    if max_quarantine_rate is not None and result.quarantine_rate > max_quarantine_rate:
        raise DataQualityError(
            f"Quarantine rate {result.quarantine_rate:.2%} exceeds tolerance "
            f"{max_quarantine_rate:.2%}; violations: {result.violation_counts()}"
        )
    return result
