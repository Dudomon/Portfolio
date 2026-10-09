# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Publish producer features (Feature Engineering in Unity Catalog)
# MAGIC
# MAGIC Upserts point-in-time producer snapshots into a time-series feature table.
# MAGIC Runs for historical applications (training) and for new applications
# MAGIC (so scoring can look up the snapshot as of each new application date).

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_credit")
dbutils.widgets.text("schema", "credit_risk")
dbutils.widgets.dropdown("dataset", "loan_applications", ["loan_applications", "new_applications"])

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

from credit_risk.config import UnityCatalogNames
from credit_risk.lakehouse.gold import publish_producer_features

names = UnityCatalogNames(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))

publish_producer_features(
    spark,
    FeatureEngineeringClient(),
    silver_table=names.silver(dbutils.widgets.get("dataset")),
    feature_table=names.producer_features,
)
