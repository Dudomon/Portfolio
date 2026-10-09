"""Feature contract and Unity Catalog naming.

The feature lists here are the single source of truth for what the model may
see. Sensitive attributes are deliberately absent from every feature list: they
are carried alongside predictions for evaluation and monitoring only. See
docs/RESPONSIBLE_AI.md for why exclusion alone is not treated as "fair".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

LABEL = "defaulted"
LOAN_KEY = "loan_id"
ENTITY_KEY = "producer_id"
TIMESTAMP_KEY = "application_date"
FEATURE_TIMESTAMP_KEY = "as_of_date"

# Used for fairness evaluation, slicing in Lakehouse Monitoring, and reweighing.
# Never passed to the estimator.
SENSITIVE_ATTRIBUTES: tuple[str, ...] = ("region", "gender", "farm_size_band")

# Producer-level features, published to the feature table and joined at
# training/scoring time with a point-in-time lookup.
PRODUCER_FEATURES: tuple[str, ...] = (
    "farm_size_ha",
    "years_farming",
    "credit_history_months",
    "prior_defaults",
    "cooperative_member",
    "irrigated",
    "revenue_per_ha",
    "debt_to_revenue",
)

# Loan-level features, computed on the application row itself.
LOAN_NUMERIC_FEATURES: tuple[str, ...] = (
    "loan_to_revenue",
    "installment_to_revenue",
    "term_months",
    "interest_rate",
    "has_crop_insurance",
    "rainfall_anomaly",
)
LOAN_CATEGORICAL_FEATURES: tuple[str, ...] = ("crop_type",)

NUMERIC_FEATURES: tuple[str, ...] = PRODUCER_FEATURES + LOAN_NUMERIC_FEATURES
CATEGORICAL_FEATURES: tuple[str, ...] = LOAN_CATEGORICAL_FEATURES
MODEL_FEATURES: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CHAMPION_ALIAS = "champion"
CHALLENGER_ALIAS = "challenger"
PREVIOUS_CHAMPION_ALIAS = "previous_champion"

# Historical applications with outcomes (training) and fresh applications
# awaiting a decision (scoring) flow through the same bronze/silver code.
Dataset = Literal["loan_applications", "new_applications"]

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _check_identifier(value: str, kind: str) -> str:
    # Names are interpolated into SQL and model URIs; rejecting anything that is
    # not a plain identifier avoids quoting bugs and injection through widgets.
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid Unity Catalog {kind} name: {value!r}")
    return value


@dataclass(frozen=True)
class UnityCatalogNames:
    """Three-level names for every table, volume, and model of one target."""

    catalog: str
    schema: str
    model_basename: str = "rural_credit_default"

    def __post_init__(self) -> None:
        _check_identifier(self.catalog, "catalog")
        _check_identifier(self.schema, "schema")
        _check_identifier(self.model_basename, "model")

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{_check_identifier(name, 'table')}"

    def bronze(self, dataset: Dataset) -> str:
        return self.table(f"bronze_{dataset}")

    def silver(self, dataset: Dataset) -> str:
        return self.table(f"silver_{dataset}")

    def quarantine(self, dataset: Dataset) -> str:
        return self.table(f"silver_{dataset}_quarantine")

    def landing_path(self, dataset: Dataset) -> str:
        return f"{self.raw_volume_path}/{dataset}"

    def checkpoint_path(self, dataset: Dataset) -> str:
        return f"{self.raw_volume_path}/_checkpoints/bronze_{dataset}"

    @property
    def producer_features(self) -> str:
        return self.table("gold_producer_features")

    @property
    def predictions(self) -> str:
        return self.table("gold_credit_risk_predictions")

    @property
    def drift_metrics(self) -> str:
        return self.table("gold_feature_drift")

    @property
    def gate_audit(self) -> str:
        return self.table("governance_promotion_audit")

    @property
    def model_name(self) -> str:
        return f"{self.catalog}.{self.schema}.{self.model_basename}"

    @property
    def raw_volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/raw"

    def model_uri(self, alias: str) -> str:
        return f"models:/{self.model_name}@{_check_identifier(alias, 'alias')}"
