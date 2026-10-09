"""One evaluation report per model, consumed by the gate, MLflow, and the model card."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from credit_risk.responsible_ai.calibration import calibration_by_group, expected_calibration_error
from credit_risk.responsible_ai.fairness import FairnessSummary, assess_all


@dataclass(frozen=True)
class PerformanceMetrics:
    roc_auc: float
    pr_auc: float
    brier: float
    ece: float
    flag_rate: float
    base_rate: float

    def as_dict(self) -> dict[str, float]:
        return {
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier": self.brier,
            "ece": self.ece,
            "flag_rate": self.flag_rate,
            "base_rate": self.base_rate,
        }


@dataclass(frozen=True)
class EvaluationReport:
    threshold: float
    n_rows: int
    performance: PerformanceMetrics
    fairness: dict[str, FairnessSummary]
    calibration_by_group: dict[str, pd.DataFrame] = field(default_factory=dict)

    def flat_metrics(self, prefix: str = "") -> dict[str, float]:
        """Flat ``name -> value`` map suitable for ``mlflow.log_metrics``."""
        metrics = {f"{prefix}{k}": v for k, v in self.performance.as_dict().items()}
        for attribute, summary in self.fairness.items():
            metrics[f"{prefix}{attribute}_dp_diff"] = summary.demographic_parity_difference
            metrics[f"{prefix}{attribute}_eo_diff"] = summary.equalized_odds_difference
            metrics[f"{prefix}{attribute}_fpr_diff"] = summary.false_positive_rate_difference
        return metrics


def evaluate(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray,
    sensitive: pd.DataFrame,
    *,
    threshold: float,
    y_pred: np.ndarray | None = None,
) -> EvaluationReport:
    """Score-based and decision-based metrics at the policy's operating point.

    ``y_pred`` overrides thresholding, used to evaluate post-processing
    benchmarks whose decisions are not a single cut on the score.
    """
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    decisions = (
        (score >= threshold).astype(int) if y_pred is None else np.asarray(y_pred, dtype=int)
    )
    performance = PerformanceMetrics(
        roc_auc=float(roc_auc_score(y, score)),
        pr_auc=float(average_precision_score(y, score)),
        brier=float(brier_score_loss(y, score)),
        ece=expected_calibration_error(y, score),
        flag_rate=float(decisions.mean()),
        base_rate=float(y.mean()),
    )
    return EvaluationReport(
        threshold=threshold,
        n_rows=len(y),
        performance=performance,
        fairness=assess_all(y, decisions, sensitive.reset_index(drop=True)),
        calibration_by_group={
            str(column): calibration_by_group(y, score, sensitive[column])
            for column in sensitive.columns
        },
    )
