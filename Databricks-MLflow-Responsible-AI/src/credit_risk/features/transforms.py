"""Gold-layer feature transforms.

Every function is row-local, so Spark runs the exact same code through
``mapInPandas``; there is no second implementation to drift out of sync.

Producer features are snapshotted per application date and published to a
time-series feature table, which lets training use a point-in-time lookup and
avoids leaking information from after the application was made.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from credit_risk.config import (
    ENTITY_KEY,
    FEATURE_TIMESTAMP_KEY,
    LOAN_KEY,
    PRODUCER_FEATURES,
    TIMESTAMP_KEY,
)


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide, returning NaN instead of +/-inf when the denominator is not positive."""
    num = pd.to_numeric(numerator, errors="coerce").astype(float)
    den = pd.to_numeric(denominator, errors="coerce").astype(float)
    return num.div(den.where(den > 0))


def annual_installment(
    principal: pd.Series, annual_rate: pd.Series, term_months: pd.Series
) -> pd.Series:
    """Annualized payment of a fully amortizing loan (French/Price table).

    Price-table amortization is the common schedule for Brazilian rural credit.
    A zero rate degenerates to straight-line repayment.
    """
    p = pd.to_numeric(principal, errors="coerce").to_numpy(dtype=float)
    r = pd.to_numeric(annual_rate, errors="coerce").to_numpy(dtype=float) / 12.0
    n = pd.to_numeric(term_months, errors="coerce").to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        amortizing = p * r / (1.0 - np.power(1.0 + r, -n))
        monthly = np.where(r > 0, amortizing, p / n)
    return pd.Series(monthly * 12.0, index=principal.index)


def producer_features(df: pd.DataFrame) -> pd.DataFrame:
    """One feature snapshot per (producer, application date)."""
    out = pd.DataFrame(
        {
            ENTITY_KEY: df[ENTITY_KEY],
            FEATURE_TIMESTAMP_KEY: df[TIMESTAMP_KEY],
            "farm_size_ha": df["farm_size_ha"].astype(float),
            "years_farming": df["years_farming"].astype("int64"),
            "credit_history_months": df["credit_history_months"].astype("int64"),
            "prior_defaults": df["prior_defaults"].astype("int64"),
            "cooperative_member": df["cooperative_member"].astype("int64"),
            "irrigated": df["irrigated"].astype("int64"),
            "revenue_per_ha": safe_ratio(df["annual_revenue_brl"], df["farm_size_ha"]),
            "debt_to_revenue": safe_ratio(df["existing_debt_brl"], df["annual_revenue_brl"]),
        },
        index=df.index,
    )
    # The same producer can file two applications on one day; the snapshot is
    # identical in that case, and the feature table's primary key must be unique.
    return out.drop_duplicates(subset=[ENTITY_KEY, FEATURE_TIMESTAMP_KEY], keep="last")


def loan_features(df: pd.DataFrame, passthrough: Sequence[str] = ()) -> pd.DataFrame:
    """Application-level features plus lookup keys and any passthrough columns.

    ``passthrough`` carries the label and sensitive attributes next to the
    features without the caller having to re-join them later.
    """
    out = pd.DataFrame(
        {
            LOAN_KEY: df[LOAN_KEY],
            ENTITY_KEY: df[ENTITY_KEY],
            TIMESTAMP_KEY: df[TIMESTAMP_KEY],
            "loan_to_revenue": safe_ratio(df["loan_amount_brl"], df["annual_revenue_brl"]),
            "installment_to_revenue": safe_ratio(
                annual_installment(df["loan_amount_brl"], df["interest_rate"], df["term_months"]),
                df["annual_revenue_brl"],
            ),
            "term_months": df["term_months"].astype("int64"),
            "interest_rate": df["interest_rate"].astype(float),
            "has_crop_insurance": df["has_crop_insurance"].astype("int64"),
            "rainfall_anomaly": df["rainfall_anomaly"].astype(float),
            "crop_type": df["crop_type"].astype("string"),
        },
        index=df.index,
    )
    for column in passthrough:
        out[column] = df[column]
    return out


def join_point_in_time(loans: pd.DataFrame, producers: pd.DataFrame) -> pd.DataFrame:
    """Local equivalent of a Feature Engineering point-in-time lookup.

    For each loan, take the latest producer snapshot at or before the
    application date, exactly as ``FeatureLookup(timestamp_lookup_key=...)``
    does on Databricks.
    """
    left = loans.assign(_ts=pd.to_datetime(loans[TIMESTAMP_KEY])).sort_values("_ts")
    right = producers.assign(_ts=pd.to_datetime(producers[FEATURE_TIMESTAMP_KEY]))
    right = right.sort_values("_ts").drop(columns=FEATURE_TIMESTAMP_KEY)
    merged = pd.merge_asof(
        left,
        right[[ENTITY_KEY, "_ts", *PRODUCER_FEATURES]],
        on="_ts",
        by=ENTITY_KEY,
        direction="backward",
    )
    return merged.drop(columns="_ts").set_index(left.index).sort_index()
