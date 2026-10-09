"""Bronze ingestion with Auto Loader and an enforced schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyspark.sql import functions as F

from credit_risk.schemas import BRONZE_DDL

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def autoloader_options(file_format: str = "csv") -> dict[str, str]:
    """Auto Loader options for landing-zone files.

    The schema is supplied explicitly (not inferred) so a malformed upstream
    file cannot silently change column types. Values that do not fit the schema
    are kept in ``_rescued_data`` instead of becoming nulls, and the silver
    rules quarantine those rows.
    """
    options = {
        "cloudFiles.format": file_format,
        "rescuedDataColumn": "_rescued_data",
        # Many small files per day; incremental listing is cheaper than
        # re-listing the whole directory on every trigger.
        "cloudFiles.useIncrementalListing": "auto",
    }
    if file_format == "csv":
        options |= {"header": "true", "dateFormat": "yyyy-MM-dd"}
    return options


def ingest_to_bronze(
    spark: SparkSession,
    *,
    source_path: str,
    checkpoint_path: str,
    target_table: str,
    file_format: str = "csv",
) -> None:
    """Incrementally load new files and stop (``availableNow``).

    A triggered stream gives exactly-once file tracking through the checkpoint
    while still running as an ordinary batch job task.
    """
    stream = (
        spark.readStream.format("cloudFiles")
        .options(**autoloader_options(file_format))
        .schema(BRONZE_DDL)
        .load(source_path)
        .select(
            "*",
            F.col("_metadata.file_path").alias("_source_file"),
            F.current_timestamp().alias("_ingested_at"),
        )
    )
    query = (
        stream.writeStream.option("checkpointLocation", checkpoint_path)
        .option("mergeSchema", "false")
        .trigger(availableNow=True)
        .toTable(target_table)
    )
    query.awaitTermination()
