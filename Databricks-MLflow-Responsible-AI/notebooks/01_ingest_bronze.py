# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Ingest bronze (Auto Loader)
# MAGIC
# MAGIC Incrementally loads new CSV files from the landing volume into a bronze Delta
# MAGIC table with an enforced schema. Non-conforming values go to `_rescued_data`.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_credit")
dbutils.widgets.text("schema", "credit_risk")
dbutils.widgets.dropdown("dataset", "loan_applications", ["loan_applications", "new_applications"])

# COMMAND ----------

from credit_risk.config import UnityCatalogNames
from credit_risk.lakehouse.bronze import ingest_to_bronze

names = UnityCatalogNames(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
dataset = dbutils.widgets.get("dataset")

ingest_to_bronze(
    spark,
    source_path=names.landing_path(dataset),
    checkpoint_path=names.checkpoint_path(dataset),
    target_table=names.bronze(dataset),
)
print(f"{names.bronze(dataset)}: {spark.read.table(names.bronze(dataset)).count():,} rows")
