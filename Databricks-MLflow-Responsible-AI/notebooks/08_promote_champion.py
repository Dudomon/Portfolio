# Databricks notebook source
# MAGIC %md
# MAGIC # 08 - Promote or roll back `@champion` (human approval)
# MAGIC
# MAGIC The production path to `@champion`. A named approver runs this job after
# MAGIC reviewing the version's model card (`model_card.md` in its source run). The
# MAGIC registry helper re-checks that the version carries a passing gate decision,
# MAGIC and every change is written to the audit table. Rollback moves `@champion`
# MAGIC back to `@previous_champion`.

# COMMAND ----------

dbutils.widgets.text("catalog", "prod_credit")
dbutils.widgets.text("schema", "credit_risk")
dbutils.widgets.dropdown("action", "promote", ["promote", "rollback"])
dbutils.widgets.text("version", "")
dbutils.widgets.text("approved_by", "")

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

from credit_risk.config import CHAMPION_ALIAS, UnityCatalogNames
from credit_risk.governance.audit import alias_change_record
from credit_risk.governance.registry import (
    get_alias_version,
    promote_to_champion,
    rollback_champion,
)
from credit_risk.lakehouse.audit import append_audit

names = UnityCatalogNames(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
action = dbutils.widgets.get("action")
approved_by = dbutils.widgets.get("approved_by").strip()
if not approved_by:
    raise ValueError("approved_by is required: promotions must name an accountable person")

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

previous = get_alias_version(client, names.model_name, CHAMPION_ALIAS)
if action == "promote":
    version = dbutils.widgets.get("version").strip()
    if not version:
        raise ValueError("version is required for promote")
    promote_to_champion(client, names.model_name, version, approved_by=approved_by)
else:
    version = rollback_champion(client, names.model_name, approved_by=approved_by)

append_audit(
    spark,
    alias_change_record(
        "promotion" if action == "promote" else "rollback",
        model_name=names.model_name,
        model_version=version,
        previous_version=previous,
        actor=approved_by,
    ),
    table=names.gate_audit,
)
print(f"{action}: @champion v{previous} -> v{version} (approved by {approved_by})")
