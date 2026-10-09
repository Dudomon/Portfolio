"""Silver build: deduplicate in Spark, clean and validate in pandas partitions."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

from credit_risk.quality.cleaning import clean_applications
from credit_risk.quality.rules import DataQualityError, annotate_violations
from credit_risk.schemas import (
    BRONZE_COLUMNS,
    DERIVED_COLUMNS,
    DQ_FAILED_RULES_COLUMN,
    INGEST_METADATA_COLUMNS,
    SILVER_ANNOTATED_DDL,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession

_ANNOTATED_COLUMNS = BRONZE_COLUMNS + INGEST_METADATA_COLUMNS + DERIVED_COLUMNS


def _conform(pdf: pd.DataFrame) -> pd.DataFrame:
    """Restore nullable integer/boolean dtypes before handing data back to Arrow.

    Spark delivers a nullable INT column with nulls as float64; returning it
    unchanged would fail the Arrow cast to the declared INT schema.
    """
    out = pdf.copy()
    for name, spark_type in _ANNOTATED_COLUMNS:
        if spark_type == "INT":
            out[name] = out[name].astype("Int32")
        elif spark_type == "BOOLEAN":
            out[name] = out[name].astype("boolean")
    return out[[name for name, _ in _ANNOTATED_COLUMNS] + [DQ_FAILED_RULES_COLUMN]]


def _annotate_partitions(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    for pdf in batches:
        yield _conform(annotate_violations(clean_applications(pdf)))


def latest_per_loan(bronze: DataFrame) -> DataFrame:
    """Latest record per ``loan_id``; rows without a key pass through to be quarantined."""
    window = Window.partitionBy("loan_id").orderBy(F.col("_ingested_at").desc())
    keyed = (
        bronze.filter(F.col("loan_id").isNotNull())
        .withColumn("_rank", F.row_number().over(window))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )
    return keyed.unionByName(bronze.filter(F.col("loan_id").isNull()))


@dataclass(frozen=True)
class SilverStats:
    total: int
    quarantined: int

    @property
    def quarantine_rate(self) -> float:
        return self.quarantined / self.total if self.total else 0.0


def build_silver(
    spark: SparkSession,
    *,
    bronze_table: str,
    silver_table: str,
    quarantine_table: str,
    max_quarantine_rate: float,
) -> SilverStats:
    """Rebuild silver and its quarantine table from bronze.

    Both tables are fully rebuilt on each run, which keeps the task idempotent
    and lets rule changes apply retroactively (a fixed rule releases rows from
    quarantine); at this data volume that is cheaper
    than maintaining a MERGE. The quality check runs *before* any write, so a
    failed batch leaves the previous silver table intact.
    """
    annotated = (
        latest_per_loan(spark.read.table(bronze_table))
        .mapInPandas(_annotate_partitions, schema=SILVER_ANNOTATED_DDL)
        .persist()
    )
    failed = F.size(F.col(DQ_FAILED_RULES_COLUMN)) > 0
    stats = SilverStats(total=annotated.count(), quarantined=annotated.filter(failed).count())
    if stats.quarantine_rate > max_quarantine_rate:
        annotated.unpersist()
        raise DataQualityError(
            f"{stats.quarantined}/{stats.total} rows failed data-quality rules "
            f"({stats.quarantine_rate:.2%} > {max_quarantine_rate:.2%}); silver not updated"
        )

    (
        annotated.filter(~failed)
        .drop(DQ_FAILED_RULES_COLUMN)
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(silver_table)
    )
    (
        annotated.filter(failed)
        .withColumn("_quarantined_at", F.current_timestamp())
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(quarantine_table)
    )
    annotated.unpersist()
    return stats
