"""Optuna hyperparameter search with an inner temporal validation split."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_risk.config import LABEL
from credit_risk.training.model import (
    build_pipeline,
    feature_matrix,
    predict_default_proba,
    temporal_split,
)

# (trial_number, params, validation_auc) -> None. Lets the caller log each
# trial as a nested MLflow run without this module importing MLflow.
TrialCallback = Callable[[int, dict[str, Any], float], None]


@dataclass(frozen=True)
class TuningResult:
    best_params: dict[str, Any]
    best_value: float
    n_trials: int


def suggest_params(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 7, 63),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def tune(
    train: pd.DataFrame,
    *,
    n_trials: int,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
    on_trial: TrialCallback | None = None,
) -> TuningResult:
    """Maximize validation ROC AUC on the most recent slice of the training data.

    The holdout set is never seen here; it is reserved for the promotion gate.
    """
    import optuna

    inner = temporal_split(train, holdout_fraction=0.2)
    fit_weight = None
    if sample_weight is not None:
        weights = pd.Series(sample_weight, index=train.index)
        fit_weight = weights.loc[inner.train.index].to_numpy()

    def objective(trial: Any) -> float:
        params = suggest_params(trial)
        pipeline = build_pipeline(params, seed=seed)
        pipeline.fit(
            feature_matrix(inner.train),
            inner.train[LABEL].astype(int),
            model__sample_weight=fit_weight,
        )
        auc = float(
            roc_auc_score(
                inner.holdout[LABEL].astype(int), predict_default_proba(pipeline, inner.holdout)
            )
        )
        if on_trial is not None:
            on_trial(trial.number, params, auc)
        return auc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return TuningResult(
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        n_trials=len(study.trials),
    )
