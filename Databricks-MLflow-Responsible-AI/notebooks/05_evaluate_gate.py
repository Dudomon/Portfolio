# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Evaluate and gate (champion vs challenger)
# MAGIC
# MAGIC Scores the same temporal holdout with the new candidate and the current
# MAGIC `@champion`, applies the versioned promotion policy (performance, calibration,
# MAGIC fairness, drift), renders the model card, and records the decision in MLflow
# MAGIC and the Delta audit table. The decision is passed to the next task; this
# MAGIC notebook never changes the registry.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_credit")
dbutils.widgets.text("schema", "credit_risk")
dbutils.widgets.text("experiment_path", "/Shared/credit-risk-rai")
dbutils.widgets.text("policy_path", "")
# Only used when running the notebook interactively, outside the job.
dbutils.widgets.text("run_id", "")

# COMMAND ----------

from datetime import UTC, datetime

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow import MlflowClient
from pyspark.sql import functions as F

from credit_risk.config import CHAMPION_ALIAS, LABEL, LOAN_KEY, TIMESTAMP_KEY, UnityCatalogNames
from credit_risk.governance.policy import load_policy
from credit_risk.governance.registry import get_alias_version
from credit_risk.lakehouse.audit import append_audit
from credit_risk.lakehouse.gold import build_training_set, training_frame
from credit_risk.lakehouse.scoring import aligned_scores, score_to_pandas
from credit_risk.pipeline import run_gate
from credit_risk.responsible_ai.model_card import ModelCardContext, render_model_card
from credit_risk.tracking import (
    load_card_inputs,
    log_evaluation,
    log_gate_decision,
    log_model_card,
)
from credit_risk.training.model import temporal_split

names = UnityCatalogNames(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
policy = load_policy(dbutils.widgets.get("policy_path"))
run_id = dbutils.jobs.taskValues.get(
    taskKey="train", key="run_id", debugValue=dbutils.widgets.get("run_id")
)
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(dbutils.widgets.get("experiment_path"))
client = MlflowClient()
fe = FeatureEngineeringClient()
actor = spark.sql("SELECT current_user()").first()[0]

# COMMAND ----------

# Rebuild the identical split the candidate was trained on (deterministic cutoff).
silver_table = names.silver("loan_applications")
training_set = build_training_set(
    spark, fe, silver_table=silver_table, feature_table=names.producer_features
)
split = temporal_split(training_frame(spark, training_set, silver_table=silver_table))
holdout_applications = spark.read.table(silver_table).filter(
    F.col(LABEL).isNotNull() & (F.col(TIMESTAMP_KEY) >= F.lit(split.cutoff))
)

challenger_scores = aligned_scores(
    score_to_pandas(fe, f"runs:/{run_id}/model", holdout_applications), split.holdout[LOAN_KEY]
)
champion_version = get_alias_version(client, names.model_name, CHAMPION_ALIAS)
champion_scores = (
    aligned_scores(
        score_to_pandas(fe, names.model_uri(CHAMPION_ALIAS), holdout_applications),
        split.holdout[LOAN_KEY],
    )
    if champion_version is not None
    else None
)
outcome = run_gate(policy, split, challenger_scores, champion_scores)

# COMMAND ----------

with mlflow.start_run(run_id=run_id):
    log_evaluation(outcome.challenger, "holdout_", "evaluation/challenger")
    if outcome.champion is not None:
        log_evaluation(outcome.champion, "champion_holdout_", "evaluation/champion")
        mlflow.set_tag("gate.champion_version", champion_version)
    mlflow.log_dict(outcome.drift_psi, "governance/drift_psi.json")
    log_gate_decision(outcome.decision)
    log_model_card(
        render_model_card(
            ModelCardContext(
                model_name=names.model_name,
                run_id=run_id,
                generated_at=datetime.now(UTC),
                inputs=load_card_inputs(run_id),
                evaluation=outcome.challenger,
                gate=outcome.decision,
            )
        )
    )

append_audit(
    spark,
    outcome.decision.to_record(
        model_name=names.model_name, run_id=run_id, champion_version=champion_version, actor=actor
    ),
    table=names.gate_audit,
)
dbutils.jobs.taskValues.set(key="gate_passed", value=outcome.decision.passed)

for check in outcome.decision.failed_checks:
    print(f"FAIL {check.name}: observed={check.observed} threshold={check.threshold}")
print(f"Gate {'PASSED' if outcome.decision.passed else 'FAILED'} (policy {policy.version})")
