# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Build silver (cleaning, data-quality rules, quarantine)
# MAGIC
# MAGIC Deduplicates bronze, normalizes codes, applies the rules in
# MAGIC `credit_risk.quality.rules`, and splits failing rows into a quarantine table
# MAGIC with the names of the rules they broke. The task fails, leaving silver
# MAGIC untouched, if the quarantine rate exceeds the tolerance.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_credit")
dbutils.widgets.text("schema", "credit_risk")
dbutils.widgets.dropdown("dataset", "loan_applications", ["loan_applications", "new_applications"])
dbutils.widgets.text("max_quarantine_rate", "0.05")

# COMMAND ----------

from credit_risk.config import UnityCatalogNames
from credit_risk.lakehouse.silver import build_silver

names = UnityCatalogNames(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
dataset = dbutils.widgets.get("dataset")

stats = build_silver(
    spark,
    bronze_table=names.bronze(dataset),
    silver_table=names.silver(dataset),
    quarantine_table=names.quarantine(dataset),
    max_quarantine_rate=float(dbutils.widgets.get("max_quarantine_rate")),
)
print(f"{stats.total:,} rows, {stats.quarantined:,} quarantined ({stats.quarantine_rate:.2%})")

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT rule, COUNT(*) AS rows
        FROM {names.quarantine(dataset)}
        LATERAL VIEW explode(_dq_failed_rules) AS rule
        GROUP BY rule ORDER BY rows DESC
        """
    )
)
