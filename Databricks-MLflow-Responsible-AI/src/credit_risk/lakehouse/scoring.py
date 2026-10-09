"""Batch scoring with Feature Engineering lookups and drift metric writes."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from pyspark.sql import functions as F

from credit_risk.config import LABEL, LOAN_KEY, SENSITIVE_ATTRIBUTES
from credit_risk.lakehouse.gold import loan_feature_frame
from credit_risk.lakehouse.monitor import (
    MODEL_ID_COLUMN,
    PREDICTED_LABEL,
    PREDICTED_PROBA,
    TIMESTAMP_COLUMN,
)
from credit_risk.monitoring.drift import drift_level

if TYPE_CHECKING:
    from databricks.feature_engineering import FeatureEngineeringClient
    from pyspark.sql import DataFrame, SparkSession

# Column Feature Engineering's score_batch adds; reused as the monitor's probability column.
PREDICTION = PREDICTED_PROBA


def score_with_lookups(
    fe: FeatureEngineeringClient,
    model_uri: str,
    applications: DataFrame,
    passthrough: tuple[str, ...] = SENSITIVE_ATTRIBUTES,
) -> DataFrame:
    """Score applications; producer features are looked up point-in-time by the model.

    ``passthrough`` columns ride along for monitoring slices; the model's
    feature spec ignores them.
    """
    return fe.score_batch(
        model_uri=model_uri,
        df=loan_feature_frame(applications, passthrough=passthrough),
        result_type="double",
    )


def score_to_pandas(
    fe: FeatureEngineeringClient, model_uri: str, applications: DataFrame
) -> pd.Series:
    """Default probability indexed by ``loan_id`` (used by the promotion gate)."""
    scored = score_with_lookups(fe, model_uri, applications, passthrough=())
    pdf = scored.select(LOAN_KEY, PREDICTION).toPandas()
    return pd.Series(pdf[PREDICTION].to_numpy(dtype=float), index=pdf[LOAN_KEY], name=PREDICTION)


def aligned_scores(scores: pd.Series, loan_ids: pd.Series) -> np.ndarray:
    """Reorder scores to match a frame's row order; a missing score is a hard error."""
    aligned = scores.reindex(loan_ids.to_numpy())
    if aligned.isna().any():
        raise ValueError(f"{int(aligned.isna().sum())} rows were not scored")
    return aligned.to_numpy(dtype=float)


def write_predictions(
    scored: DataFrame,
    *,
    table: str,
    model_name: str,
    model_version: str,
    decision_threshold: float,
    policy_version: str,
) -> None:
    """Append to the inference table monitored by Lakehouse Monitoring.

    The label column is written as null and back-filled when loans mature, which
    is what lets the monitor compute performance and fairness by slice later.
    """
    (
        scored.withColumn("model_name", F.lit(model_name))
        .withColumn(MODEL_ID_COLUMN, F.lit(model_version))
        .withColumn("policy_version", F.lit(policy_version))
        .withColumn(PREDICTED_LABEL, (F.col(PREDICTION) >= F.lit(decision_threshold)).cast("int"))
        .withColumn(TIMESTAMP_COLUMN, F.current_timestamp())
        .withColumn(LABEL, F.lit(None).cast("int"))
        .write.mode("append")
        .option("mergeSchema", "true")
        .saveAsTable(table)
    )


def sample_to_pandas(df: DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """Bounded random sample for PSI; bin proportions are stable well below 50k rows."""
    total = df.count()
    fraction = min(1.0, max_rows / total) if total else 1.0
    sample: pd.DataFrame = df.sample(withReplacement=False, fraction=fraction, seed=seed).toPandas()
    return sample


def write_drift_metrics(
    spark: SparkSession, psi: Mapping[str, float], *, table: str, model_version: str
) -> None:
    computed_at = datetime.now(UTC)
    rows = [
        (computed_at, model_version, feature, float(value), drift_level(value))
        for feature, value in psi.items()
    ]
    schema = "computed_at TIMESTAMP, model_version STRING, feature STRING, psi DOUBLE, level STRING"
    spark.createDataFrame(rows, schema=schema).write.mode("append").saveAsTable(table)
