"""Stage functions shared by the Databricks notebooks and the local runner.

Each function takes and returns pandas objects. Notebooks are responsible for
moving data between Delta and pandas (via Spark / Feature Engineering) and for
MLflow side effects; the logic in between lives here once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.config import (
    CATEGORICAL_FEATURES,
    LABEL,
    NUMERIC_FEATURES,
    SENSITIVE_ATTRIBUTES,
)
from credit_risk.evaluation import EvaluationReport, evaluate
from credit_risk.features.transforms import join_point_in_time, loan_features, producer_features
from credit_risk.governance.gate import GateDecision, evaluate_gate
from credit_risk.governance.policy import PromotionPolicy
from credit_risk.monitoring.drift import psi_report
from credit_risk.quality.cleaning import clean_applications, deduplicate_latest
from credit_risk.quality.rules import ValidationResult, annotate_violations, split_valid
from credit_risk.responsible_ai.mitigation import (
    reweighing_weights,
    threshold_optimizer_benchmark,
)
from credit_risk.responsible_ai.model_card import MitigationComparison, TrainingDataSummary
from credit_risk.training.model import (
    TemporalSplit,
    build_pipeline,
    feature_matrix,
    predict_default_proba,
)
from credit_risk.training.tuning import TrialCallback, TuningResult, tune

Mitigation = Literal["none", "reweighing"]


def prepare_silver(raw: pd.DataFrame, max_quarantine_rate: float | None = 0.05) -> ValidationResult:
    """Bronze -> silver: deduplicate, clean, validate, split off quarantine."""
    return split_valid(
        annotate_violations(clean_applications(deduplicate_latest(raw))),
        max_quarantine_rate=max_quarantine_rate,
    )


def build_model_frame(silver: pd.DataFrame) -> pd.DataFrame:
    """Loan features + point-in-time producer features + label + sensitive attributes."""
    loans = loan_features(silver, passthrough=(LABEL, *SENSITIVE_ATTRIBUTES))
    return join_point_in_time(loans, producer_features(silver))


def labeled(frame: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known outcome; immature loans cannot be used for training."""
    out = frame.loc[frame[LABEL].notna()].copy()
    out[LABEL] = out[LABEL].astype(int)
    return out


@dataclass(frozen=True)
class TrainedCandidate:
    pipeline: Pipeline
    params: dict[str, Any]
    mitigation: Mitigation
    tuning: TuningResult | None


def sample_weights(train: pd.DataFrame, mitigation: Mitigation) -> np.ndarray | None:
    if mitigation == "none":
        return None
    return reweighing_weights(train[LABEL], train[list(SENSITIVE_ATTRIBUTES)])


def fit_pipeline(
    train: pd.DataFrame, params: dict[str, Any], weights: np.ndarray | None, seed: int
) -> Pipeline:
    pipeline = build_pipeline(params, seed=seed)
    pipeline.fit(feature_matrix(train), train[LABEL].astype(int), model__sample_weight=weights)
    return pipeline


def fit_candidate(
    train: pd.DataFrame,
    *,
    mitigation: Mitigation,
    n_trials: int,
    seed: int = 42,
    on_trial: TrialCallback | None = None,
) -> TrainedCandidate:
    """Tune (optionally) and fit the candidate with the configured mitigation."""
    weights = sample_weights(train, mitigation)
    tuning = (
        tune(train, n_trials=n_trials, sample_weight=weights, seed=seed, on_trial=on_trial)
        if n_trials > 0
        else None
    )
    params = dict(tuning.best_params) if tuning else {}
    return TrainedCandidate(
        pipeline=fit_pipeline(train, params, weights, seed),
        params=params,
        mitigation=mitigation,
        tuning=tuning,
    )


def evaluate_scores(
    holdout: pd.DataFrame, scores: np.ndarray, threshold: float
) -> EvaluationReport:
    return evaluate(
        holdout[LABEL],
        scores,
        holdout[list(SENSITIVE_ATTRIBUTES)],
        threshold=threshold,
    )


def compare_mitigations(
    split: TemporalSplit, candidate: TrainedCandidate, threshold: float, seed: int = 42
) -> tuple[MitigationComparison, ...]:
    """Baseline vs reweighing (both deployable) vs ThresholdOptimizer (benchmark).

    Both deployable variants reuse the candidate's hyperparameters so the
    comparison isolates the effect of the mitigation.
    """
    holdout = split.holdout
    reports: dict[Mitigation, EvaluationReport] = {}
    scores: dict[Mitigation, np.ndarray] = {}
    for variant in ("none", "reweighing"):
        pipeline = (
            candidate.pipeline
            if variant == candidate.mitigation
            else fit_pipeline(
                split.train, candidate.params, sample_weights(split.train, variant), seed
            )
        )
        scores[variant] = predict_default_proba(pipeline, holdout)
        reports[variant] = evaluate_scores(holdout, scores[variant], threshold)

    baseline = reports["none"]
    worst_attribute = max(
        baseline.fairness, key=lambda a: baseline.fairness[a].equalized_odds_difference
    )
    benchmark = threshold_optimizer_benchmark(
        holdout[LABEL], scores["none"], holdout[worst_attribute], seed=seed
    )
    benchmark_report = evaluate(
        holdout[LABEL],
        scores["none"],
        holdout[list(SENSITIVE_ATTRIBUTES)],
        threshold=threshold,
        y_pred=benchmark.y_pred,
    )
    return (
        MitigationComparison.from_report(
            "baseline", baseline, deployable=True, note="no mitigation"
        ),
        MitigationComparison.from_report(
            "reweighing",
            reports["reweighing"],
            deployable=True,
            note="sensitive attributes used only to weight training rows",
        ),
        MitigationComparison.from_report(
            f"threshold_optimizer[{worst_attribute}]",
            benchmark_report,
            deployable=False,
            note=(
                f"needs {worst_attribute} at decision time; benchmark only "
                "(AUC shown is the underlying baseline score)"
            ),
        ),
    )


def training_summary(split: TemporalSplit, source: str) -> TrainingDataSummary:
    return TrainingDataSummary(
        source=source,
        cutoff=split.cutoff,
        n_train=len(split.train),
        n_holdout=len(split.holdout),
        train_default_rate=float(split.train[LABEL].mean()),
        holdout_default_rate=float(split.holdout[LABEL].mean()),
    )


def drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> dict[str, float]:
    return psi_report(reference, current, NUMERIC_FEATURES, CATEGORICAL_FEATURES)


@dataclass(frozen=True)
class GateOutcome:
    challenger: EvaluationReport
    champion: EvaluationReport | None
    drift_psi: dict[str, float]
    decision: GateDecision


def run_gate(
    policy: PromotionPolicy,
    split: TemporalSplit,
    challenger_scores: np.ndarray,
    champion_scores: np.ndarray | None,
) -> GateOutcome:
    """Evaluate challenger and champion on the same holdout and apply the policy."""
    threshold = policy.decision_threshold
    challenger = evaluate_scores(split.holdout, challenger_scores, threshold)
    champion = (
        evaluate_scores(split.holdout, champion_scores, threshold)
        if champion_scores is not None
        else None
    )
    drift = drift_report(split.train, split.holdout)
    return GateOutcome(
        challenger=challenger,
        champion=champion,
        drift_psi=drift,
        decision=evaluate_gate(policy, challenger, champion, drift),
    )
