"""MLflow logging helpers shared by notebooks and the local runner.

Artifacts are logged with predictable paths (``fairness/``, ``explanations/``,
``governance/``) so reviewers and downstream jobs can find them without
reading code.
"""

from __future__ import annotations

import io
import json
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import mlflow
import pandas as pd

from credit_risk.evaluation import EvaluationReport
from credit_risk.governance.gate import GateDecision
from credit_risk.responsible_ai.explain import Explanations
from credit_risk.responsible_ai.model_card import (
    CardInputs,
    MitigationComparison,
    TrainingDataSummary,
)
from credit_risk.training.tuning import TrialCallback


def nested_trial_logger(parent_tags: Mapping[str, str] | None = None) -> TrialCallback:
    """Log each Optuna trial as a nested run under the active training run."""

    def log_trial(number: int, params: dict[str, Any], value: float) -> None:
        with mlflow.start_run(run_name=f"trial-{number:03d}", nested=True):
            mlflow.set_tags({"stage": "tuning", **(parent_tags or {})})
            mlflow.log_params(params)
            mlflow.log_metric("validation_roc_auc", value)

    return log_trial


def log_frame(df: pd.DataFrame, artifact_file: str) -> None:
    mlflow.log_text(df.to_csv(index=True), artifact_file)


def log_evaluation(report: EvaluationReport, prefix: str, artifact_dir: str) -> None:
    """Metrics for comparison in the MLflow UI; per-group tables as artifacts."""
    mlflow.log_metrics(report.flat_metrics(prefix))
    for attribute, summary in report.fairness.items():
        log_frame(summary.by_group, f"{artifact_dir}/fairness_by_{attribute}.csv")
        log_frame(
            report.calibration_by_group[attribute], f"{artifact_dir}/calibration_by_{attribute}.csv"
        )
    mlflow.log_dict(
        {attr: s.to_dict() for attr, s in report.fairness.items()},
        f"{artifact_dir}/fairness_summary.json",
    )


def log_explanations(explanations: Explanations) -> None:
    mlflow.log_text(
        json.dumps(explanations.local, indent=2, default=str),
        "explanations/local_reason_codes.json",
    )


_GATE_DECISION = "governance/gate_decision.json"


def log_gate_decision(decision: GateDecision) -> None:
    mlflow.set_tags(
        {
            "gate.passed": str(decision.passed).lower(),
            "gate.policy_version": decision.policy_version,
        }
    )
    mlflow.log_metric("gate_passed", float(decision.passed))
    mlflow.log_dict(decision.to_dict(), _GATE_DECISION)


def load_gate_decision(run_id: str) -> GateDecision:
    return GateDecision.from_dict(mlflow.artifacts.load_dict(f"runs:/{run_id}/{_GATE_DECISION}"))


def log_model_card(markdown: str) -> None:
    mlflow.log_text(markdown, "model_card.md")


_SUMMARY = "governance/training_summary.json"
_PARAMS = "governance/hyperparameters.json"
_MITIGATIONS = "governance/mitigation_comparison.json"
_IMPORTANCE = "explanations/global_importance.csv"


def log_card_inputs(inputs: CardInputs) -> None:
    """Persist card inputs on the training run so a later job task can read them."""
    mlflow.log_dict(inputs.training_data.to_dict(), _SUMMARY)
    mlflow.log_dict(inputs.hyperparameters, _PARAMS)
    mlflow.log_dict({"variants": [asdict(m) for m in inputs.mitigations]}, _MITIGATIONS)
    log_frame(inputs.global_importance.set_index("feature"), _IMPORTANCE)


def load_card_inputs(run_id: str) -> CardInputs:
    base = f"runs:/{run_id}"
    importance = pd.read_csv(io.StringIO(mlflow.artifacts.load_text(f"{base}/{_IMPORTANCE}")))
    return CardInputs(
        training_data=TrainingDataSummary.from_dict(
            mlflow.artifacts.load_dict(f"{base}/{_SUMMARY}")
        ),
        hyperparameters=dict(mlflow.artifacts.load_dict(f"{base}/{_PARAMS}")),
        mitigations=tuple(
            MitigationComparison(**row)
            for row in mlflow.artifacts.load_dict(f"{base}/{_MITIGATIONS}")["variants"]
        ),
        global_importance=importance,
    )
