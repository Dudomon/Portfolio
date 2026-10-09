"""Append governance events to the Delta audit table."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from credit_risk.governance.audit import AUDIT_COLUMNS, AUDIT_DDL, normalize

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def append_audit(spark: SparkSession, record: Mapping[str, Any], *, table: str) -> None:
    """Append one event. The table is append-only by convention and by grants:
    jobs get MODIFY, nobody gets DELETE outside the governance group."""
    row = normalize(record)
    values = tuple(row[name] for name, _ in AUDIT_COLUMNS)
    spark.createDataFrame([values], schema=AUDIT_DDL).write.mode("append").saveAsTable(table)
