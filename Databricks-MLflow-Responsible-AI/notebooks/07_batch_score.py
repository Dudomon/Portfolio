# Databricks notebook source
# MAGIC %md
# MAGIC # 07 - Batch score new applications and monitor drift
# MAGIC
# MAGIC Scores new applications with `@champion` (features looked up point-in-time),
# MAGIC appends to the inference table, writes PSI drift metrics, and ensures the
# MAGIC Lakehouse Monitoring inference profile exists and is refreshed.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_credit")
dbutils.widgets.text("schema", "credit_risk")
dbutils.widgets.text("policy_path", "")
dbutils.widgets.text("monitor_assets_dir", "")

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
from mlflow import MlflowClient

from credit_risk.config import CHAMPION_ALIAS, UnityCatalogNames
from credit_risk.governance.policy import load_policy
from credit_risk.governance.registry import get_alias_version
from credit_risk.lakehouse.monitor import ensure_inference_monitor
from credit_risk.lakehouse.scoring import (
    sample_to_pandas,
    score_with_lookups,
    write_drift_metrics,
    write_predictions,
)
from credit_risk.monitoring.drift import drift_level
from credit_risk.pipeline import build_model_frame, drift_report

names = UnityCatalogNames(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
policy = load_policy(dbutils.widgets.get("policy_path"))
mlflow.set_registry_uri("databricks-uc")
champion_version = get_alias_version(MlflowClient(), names.model_name, CHAMPION_ALIAS)
if champion_version is None:
    raise RuntimeError(f"{names.model_name} has no @{CHAMPION_ALIAS}; nothing to score with")

# COMMAND ----------

applications = spark.read.table(names.silver("new_applications"))
scored = score_with_lookups(
    FeatureEngineeringClient(), names.model_uri(CHAMPION_ALIAS), applications
)
write_predictions(
    scored,
    table=names.predictions,
    model_name=names.model_name,
    model_version=champion_version,
    decision_threshold=policy.decision_threshold,
    policy_version=policy.version,
)

# COMMAND ----------

# Reference population: historical applications the model family was built on.
reference = build_model_frame(
    sample_to_pandas(spark.read.table(names.silver("loan_applications")), max_rows=50_000)
)
current = build_model_frame(sample_to_pandas(applications, max_rows=50_000))
psi = drift_report(reference, current)
write_drift_metrics(spark, psi, table=names.drift_metrics, model_version=champion_version)
for feature, value in sorted(psi.items(), key=lambda item: -item[1]):
    print(f"{feature:<26} PSI={value:.3f} ({drift_level(value)})")

# COMMAND ----------

ensure_inference_monitor(
    WorkspaceClient(),
    table_name=names.predictions,
    output_schema=f"{names.catalog}.{names.schema}",
    assets_dir=dbutils.widgets.get("monitor_assets_dir"),
)
