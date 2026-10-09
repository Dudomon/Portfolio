# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Train candidate
# MAGIC
# MAGIC Builds a point-in-time training set with Feature Engineering, tunes LightGBM
# MAGIC with Optuna (each trial a nested MLflow run), fits the candidate with the
# MAGIC configured bias mitigation, and logs the model with its feature lookups,
# MAGIC SHAP explanations, and the mitigation comparison. Registration happens later,
# MAGIC and only if the promotion gate passes.

# COMMAND ----------

dbutils.widgets.text("catalog", "dev_credit")
dbutils.widgets.text("schema", "credit_risk")
dbutils.widgets.text("experiment_path", "/Shared/credit-risk-rai")
dbutils.widgets.dropdown("mitigation", "reweighing", ["none", "reweighing"])
dbutils.widgets.text("n_trials", "25")
dbutils.widgets.text("policy_path", "")
dbutils.widgets.text("seed", "42")

# COMMAND ----------

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient

from credit_risk.config import LOAN_KEY, UnityCatalogNames
from credit_risk.governance.policy import load_policy
from credit_risk.lakehouse.gold import build_training_set, training_frame
from credit_risk.pipeline import compare_mitigations, fit_candidate, training_summary
from credit_risk.responsible_ai.explain import explain
from credit_risk.responsible_ai.model_card import CardInputs
from credit_risk.tracking import log_card_inputs, log_explanations, nested_trial_logger
from credit_risk.training.model import temporal_split
from credit_risk.training.pyfunc import DefaultProbabilityModel, package_code_path

names = UnityCatalogNames(dbutils.widgets.get("catalog"), dbutils.widgets.get("schema"))
policy = load_policy(dbutils.widgets.get("policy_path"))
mitigation = dbutils.widgets.get("mitigation")
seed = int(dbutils.widgets.get("seed"))

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(dbutils.widgets.get("experiment_path"))
fe = FeatureEngineeringClient()

# COMMAND ----------

training_set = build_training_set(
    spark,
    fe,
    silver_table=names.silver("loan_applications"),
    feature_table=names.producer_features,
)
split = temporal_split(
    training_frame(spark, training_set, silver_table=names.silver("loan_applications"))
)
print(f"train={len(split.train):,} holdout={len(split.holdout):,} cutoff={split.cutoff}")

# COMMAND ----------

with mlflow.start_run(run_name=f"train-{mitigation}") as run:
    mlflow.set_tags({"stage": "training", "policy_version": policy.version})
    mlflow.log_params(
        {
            "mitigation": mitigation,
            "holdout_cutoff": split.cutoff.isoformat(),
            "training_table": names.silver("loan_applications"),
            "feature_table": names.producer_features,
        }
    )
    candidate = fit_candidate(
        split.train,
        mitigation=mitigation,
        n_trials=int(dbutils.widgets.get("n_trials")),
        seed=seed,
        on_trial=nested_trial_logger(),
    )
    mlflow.log_params({f"model.{k}": v for k, v in candidate.params.items()})
    if candidate.tuning is not None:
        mlflow.log_metric("best_validation_roc_auc", candidate.tuning.best_value)

    # Logging through Feature Engineering stores the lookup spec with the model,
    # so batch scoring and serving fetch producer features by key automatically.
    fe.log_model(
        model=DefaultProbabilityModel(candidate.pipeline),
        artifact_path="model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        infer_input_example=True,
        code_paths=[package_code_path()],
    )

    explanations = explain(candidate.pipeline, split.holdout, split.holdout[LOAN_KEY])
    log_explanations(explanations)
    log_card_inputs(
        CardInputs(
            training_data=training_summary(split, source=names.silver("loan_applications")),
            hyperparameters=candidate.params,
            mitigations=compare_mitigations(split, candidate, policy.decision_threshold, seed),
            global_importance=explanations.global_importance,
        )
    )

dbutils.jobs.taskValues.set(key="run_id", value=run.info.run_id)
print(f"Logged candidate in run {run.info.run_id}")
