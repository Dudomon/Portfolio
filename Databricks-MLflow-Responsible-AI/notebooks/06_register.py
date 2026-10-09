# Databricks notebook source
# MAGIC %md
# MAGIC # 06 - Register challenger (and auto-promote outside production)
# MAGIC
# MAGIC Runs only when the gate passed (job condition task). Registers the candidate
# MAGIC in Unity Catalog, tags it with the gate result, and points `@challenger` at it.
# MAGIC In dev and staging `auto_promote=true` also moves `@champion`; in production a
# MAGIC named human promotes through notebook 08 after reading the model card.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_credit")
dbutils.widgets.text("schema", "credit_risk")
dbutils.widgets.dropdown("auto_promote", "false", ["true", "false"])
dbutils.widgets.text("run_id", "")

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

from credit_risk.config import UnityCatalogNames
from credit_risk.governance.audit import alias_change_record
from credit_risk.governance.registry import promote_to_champion, register_challenger
from credit_risk.lakehouse.audit import append_audit
from credit_risk.tracking import load_gate_decision

names = UnityCatalogNames(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
run_id = dbutils.jobs.taskValues.get(
    taskKey="train", key="run_id", debugValue=dbutils.widgets.get("run_id")
)
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
actor = spark.sql("SELECT current_user()").first()[0]

# COMMAND ----------

# The decision is re-read from the run, not trusted from the job graph: the
# registry helper refuses anything without a passing, logged decision.
decision = load_gate_decision(run_id)
version = register_challenger(client, names.model_name, f"runs:/{run_id}/model", run_id, decision)
append_audit(
    spark,
    alias_change_record(
        "registration",
        model_name=names.model_name,
        model_version=version,
        previous_version=None,
        actor=actor,
    ),
    table=names.gate_audit,
)
print(f"Registered {names.model_name} v{version} as @challenger")

# COMMAND ----------

if dbutils.widgets.get("auto_promote") == "true":
    previous = promote_to_champion(client, names.model_name, version, approved_by=actor)
    append_audit(
        spark,
        alias_change_record(
            "promotion",
            model_name=names.model_name,
            model_version=version,
            previous_version=previous,
            actor=actor,
        ),
        table=names.gate_audit,
    )
    print(f"@champion: v{previous} -> v{version}")
else:
    print("auto_promote=false: awaiting human approval (notebook 08)")
