"""Row-wise normalization applied before validation.

Cleaning only fixes defects whose correct value is unambiguous (casing,
whitespace, known synonyms). Anything that would require guessing is left
untouched so the data-quality rules quarantine it instead.

All functions are row-local so they can run inside Spark ``mapInPandas``
partitions without changing semantics.
"""

from __future__ import annotations

import pandas as pd

from credit_risk.domain import REGIONS

_REGION_LOOKUP = {r.lower(): r for r in REGIONS}
_GENDER_LOOKUP = {
    "f": "F",
    "female": "F",
    "feminino": "F",
    "m": "M",
    "male": "M",
    "masculino": "M",
}

# Simplified size bands. Brazilian law defines small/medium properties in
# "fiscal modules" whose hectare size varies by municipality (5-110 ha); fixed
# hectare cut-offs are a deliberate simplification for this synthetic dataset.
FARM_SIZE_BANDS: tuple[tuple[str, float], ...] = (
    ("smallholder", 50.0),
    ("medium", 500.0),
    ("large", float("inf")),
)


def _normalize_token(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower()


def farm_size_band(farm_size_ha: pd.Series) -> pd.Series:
    """Map hectares to a size band (lower bound inclusive); non-positive or missing stay null."""
    values = pd.to_numeric(farm_size_ha, errors="coerce").astype(float)
    edges = [0.0] + [upper for _, upper in FARM_SIZE_BANDS]
    bands = pd.cut(
        values.where(values > 0),
        bins=edges,
        labels=[name for name, _ in FARM_SIZE_BANDS],
        right=False,
    )
    return bands.astype("string")


def clean_applications(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical codes and derive ``farm_size_band``."""
    out = df.copy()
    region = _normalize_token(out["region"])
    out["region"] = region.map(_REGION_LOOKUP).fillna(out["region"].astype("string"))
    gender = _normalize_token(out["gender"])
    out["gender"] = gender.map(_GENDER_LOOKUP).fillna(out["gender"].astype("string"))
    out["crop_type"] = _normalize_token(out["crop_type"])
    out["state"] = out["state"].astype("string").str.strip().str.upper()
    out["farm_size_band"] = farm_size_band(out["farm_size_ha"])
    return out


def deduplicate_latest(df: pd.DataFrame, order_by: str = "_ingested_at") -> pd.DataFrame:
    """Keep the most recently ingested record per ``loan_id``.

    Source systems re-send corrected records; the latest one wins. Spark applies
    the same rule with a window function because deduplication is not
    partition-local.
    """
    if order_by not in df.columns:
        return df.drop_duplicates(subset="loan_id", keep="last")
    return df.sort_values(order_by, kind="stable").drop_duplicates(subset="loan_id", keep="last")
