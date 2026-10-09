"""Estimator pipeline and temporal train/holdout split."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from credit_risk.config import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TIMESTAMP_KEY

# Conservative defaults: shallow trees and strong regularization suit a tabular
# credit problem with a few thousand positives and keep SHAP explanations stable.
DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}


def build_pipeline(params: Mapping[str, Any] | None = None, seed: int = 42) -> Pipeline:
    """Preprocessing + LightGBM classifier as one MLflow-loggable sklearn pipeline.

    LightGBM handles missing numeric values natively, so numeric columns pass
    through untouched; only ``crop_type`` needs encoding. Unknown crops at
    scoring time map to an all-zeros vector instead of failing the batch.
    """
    from lightgbm import LGBMClassifier  # heavy import kept local to the training path

    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    classifier = LGBMClassifier(
        **{**DEFAULT_PARAMS, **(params or {})},
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", classifier)])


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Select model inputs in contract order; extra columns are dropped on purpose.

    Numeric inputs are cast to float so the estimator always sees one dtype,
    whether rows came from pandas (int64) or Spark (nullable Int64/object).
    """
    out = df.loc[:, list(MODEL_FEATURES)].copy()
    out[list(NUMERIC_FEATURES)] = out[list(NUMERIC_FEATURES)].astype(float)
    return out


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    holdout: pd.DataFrame
    cutoff: date


def temporal_split(df: pd.DataFrame, holdout_fraction: float = 0.2) -> TemporalSplit:
    """Hold out the most recent applications.

    Credit models are used on future applicants, so a random split would
    overstate performance and hide population drift. The cutoff is the date
    quantile; every row on the cutoff date goes to the holdout so the split is
    deterministic.
    """
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be in (0, 1)")
    dates = pd.to_datetime(df[TIMESTAMP_KEY])
    # Order statistic at the (1 - fraction) position, truncated to midnight.
    position = int(len(dates) * (1 - holdout_fraction))
    cutoff = pd.Timestamp(dates.sort_values().iloc[position]).normalize()
    is_holdout = (dates >= cutoff).to_numpy()
    return TemporalSplit(
        train=df.loc[~is_holdout],
        holdout=df.loc[is_holdout],
        cutoff=cutoff.date(),
    )


def predict_default_proba(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    return np.asarray(pipeline.predict_proba(feature_matrix(df))[:, 1], dtype=float)
