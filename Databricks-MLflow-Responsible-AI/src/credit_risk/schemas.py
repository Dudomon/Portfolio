"""Table schemas expressed as Spark DDL strings.

DDL strings (rather than ``StructType`` objects) keep this module importable
without PySpark: Auto Loader's ``.schema()`` and ``mapInPandas`` both accept
them directly.
"""

from __future__ import annotations

BRONZE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("loan_id", "STRING"),
    ("producer_id", "STRING"),
    ("application_date", "DATE"),
    ("region", "STRING"),
    ("state", "STRING"),
    ("gender", "STRING"),
    ("farm_size_ha", "DOUBLE"),
    ("crop_type", "STRING"),
    ("loan_amount_brl", "DOUBLE"),
    ("term_months", "INT"),
    ("interest_rate", "DOUBLE"),
    ("annual_revenue_brl", "DOUBLE"),
    ("existing_debt_brl", "DOUBLE"),
    ("years_farming", "INT"),
    ("credit_history_months", "INT"),
    ("prior_defaults", "INT"),
    ("has_crop_insurance", "BOOLEAN"),
    ("irrigated", "BOOLEAN"),
    ("cooperative_member", "BOOLEAN"),
    ("rainfall_anomaly", "DOUBLE"),
    ("defaulted", "INT"),
)

# Added by ingestion: Auto Loader's rescued-data column plus lineage metadata.
INGEST_METADATA_COLUMNS: tuple[tuple[str, str], ...] = (
    ("_rescued_data", "STRING"),
    ("_source_file", "STRING"),
    ("_ingested_at", "TIMESTAMP"),
)

# Added by silver cleaning.
DERIVED_COLUMNS: tuple[tuple[str, str], ...] = (("farm_size_band", "STRING"),)

DQ_FAILED_RULES_COLUMN = "_dq_failed_rules"

# Feature transforms emit pandas int64, hence BIGINT rather than INT.
PRODUCER_FEATURE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("producer_id", "STRING"),
    ("as_of_date", "DATE"),
    ("farm_size_ha", "DOUBLE"),
    ("years_farming", "BIGINT"),
    ("credit_history_months", "BIGINT"),
    ("prior_defaults", "BIGINT"),
    ("cooperative_member", "BIGINT"),
    ("irrigated", "BIGINT"),
    ("revenue_per_ha", "DOUBLE"),
    ("debt_to_revenue", "DOUBLE"),
)

LOAN_FEATURE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("loan_id", "STRING"),
    ("producer_id", "STRING"),
    ("application_date", "DATE"),
    ("loan_to_revenue", "DOUBLE"),
    ("installment_to_revenue", "DOUBLE"),
    ("term_months", "BIGINT"),
    ("interest_rate", "DOUBLE"),
    ("has_crop_insurance", "BIGINT"),
    ("rainfall_anomaly", "DOUBLE"),
    ("crop_type", "STRING"),
)


COLUMN_TYPES: dict[str, str] = dict(BRONZE_COLUMNS + INGEST_METADATA_COLUMNS + DERIVED_COLUMNS)


def to_ddl(columns: tuple[tuple[str, str], ...]) -> str:
    return ", ".join(f"{name} {dtype}" for name, dtype in columns)


BRONZE_DDL = to_ddl(BRONZE_COLUMNS)
SILVER_ANNOTATED_DDL = to_ddl(
    BRONZE_COLUMNS
    + INGEST_METADATA_COLUMNS
    + DERIVED_COLUMNS
    + ((DQ_FAILED_RULES_COLUMN, "ARRAY<STRING>"),)
)
PRODUCER_FEATURES_DDL = to_ddl(PRODUCER_FEATURE_COLUMNS)
