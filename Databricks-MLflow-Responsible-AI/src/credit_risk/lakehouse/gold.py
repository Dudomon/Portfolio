"""Gold layer: producer feature table and Feature Engineering training sets."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd
from pyspark.sql import functions as F

from credit_risk.config import (
    ENTITY_KEY,
    FEATURE_TIMESTAMP_KEY,
    LABEL,
    LOAN_KEY,
    PRODUCER_FEATURES,
    SENSITIVE_ATTRIBUTES,
    TIMESTAMP_KEY,
)
from credit_risk.features.transforms import loan_features, producer_features
from credit_risk.schemas import (
    COLUMN_TYPES,
    LOAN_FEATURE_COLUMNS,
    PRODUCER_FEATURE_COLUMNS,
    PRODUCER_FEATURES_DDL,
    to_ddl,
)

if TYPE_CHECKING:
    from databricks.feature_engineering import FeatureEngineeringClient
    from databricks.feature_engineering.training_set import TrainingSet
    from pyspark.sql import DataFrame, SparkSession


def _producer_partitions(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    columns = [name for name, _ in PRODUCER_FEATURE_COLUMNS]
    for pdf in batches:
        yield producer_features(pdf)[columns]


def publish_producer_features(
    spark: SparkSession, fe: FeatureEngineeringClient, *, silver_table: str, feature_table: str
) -> None:
    """Upsert point-in-time producer snapshots into a time-series feature table.

    ``as_of_date`` is declared as a timestamp key, so training lookups return the
    snapshot valid at each application date and never a later one (no leakage).
    """
    features = (
        spark.read.table(silver_table)
        .mapInPandas(_producer_partitions, schema=PRODUCER_FEATURES_DDL)
        # Per-partition dedup cannot see duplicates split across partitions.
        .dropDuplicates([ENTITY_KEY, FEATURE_TIMESTAMP_KEY])
    )
    if not spark.catalog.tableExists(feature_table):
        fe.create_table(
            name=feature_table,
            primary_keys=[ENTITY_KEY, FEATURE_TIMESTAMP_KEY],
            timestamp_keys=[FEATURE_TIMESTAMP_KEY],
            schema=features.schema,
            description=(
                "Producer-level credit features snapshotted per application date. "
                "Synthetic data. Sensitive attributes are intentionally not stored here."
            ),
        )
    fe.write_table(name=feature_table, df=features, mode="merge")


def loan_feature_frame(applications: DataFrame, passthrough: Sequence[str] = ()) -> DataFrame:
    """Application-level features via the same pandas transform used in tests."""
    columns = LOAN_FEATURE_COLUMNS + tuple((c, COLUMN_TYPES[c]) for c in passthrough)
    names = [name for name, _ in columns]

    def transform(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in batches:
            yield loan_features(pdf, passthrough)[names]

    return applications.mapInPandas(transform, schema=to_ddl(columns))


def producer_lookups(feature_table: str) -> list[Any]:
    from databricks.feature_engineering import FeatureLookup

    return [
        FeatureLookup(
            table_name=feature_table,
            lookup_key=ENTITY_KEY,
            timestamp_lookup_key=TIMESTAMP_KEY,
            feature_names=list(PRODUCER_FEATURES),
        )
    ]


def build_training_set(
    spark: SparkSession, fe: FeatureEngineeringClient, *, silver_table: str, feature_table: str
) -> TrainingSet:
    """Labeled loans joined point-in-time to producer features.

    Sensitive attributes are not part of the training set, so they cannot end
    up in the logged model's input schema; :func:`training_frame` re-attaches
    them by ``loan_id`` for evaluation only.
    """
    labeled = spark.read.table(silver_table).filter(F.col(LABEL).isNotNull())
    return fe.create_training_set(
        df=loan_feature_frame(labeled, passthrough=(LABEL,)),
        feature_lookups=producer_lookups(feature_table),
        label=LABEL,
    )


def training_frame(
    spark: SparkSession, training_set: TrainingSet, *, silver_table: str
) -> pd.DataFrame:
    """Collect the training set plus sensitive attributes to the driver.

    A few hundred thousand rows fit comfortably in driver memory and LightGBM
    trains single-node; beyond that, switch to distributed training instead of
    collecting.
    """
    sensitive = spark.read.table(silver_table).select(LOAN_KEY, *SENSITIVE_ATTRIBUTES)
    frame: pd.DataFrame = (
        training_set.load_df().join(sensitive, on=LOAN_KEY, how="inner").toPandas()
    )
    frame[LABEL] = frame[LABEL].astype(int)
    return frame.sort_values([TIMESTAMP_KEY, LOAN_KEY]).reset_index(drop=True)
