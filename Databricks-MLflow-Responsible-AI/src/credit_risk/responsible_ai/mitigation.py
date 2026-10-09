"""Bias mitigation options.

Two families are implemented and compared:

* **Reweighing** (pre-processing, Kamiran & Calders 2012). Training rows are
  weighted so that group membership and label are statistically independent in
  the weighted data. Sensitive attributes are needed only at training time,
  so the deployed model still never receives them. This is the deployable
  option.
* **ThresholdOptimizer** (post-processing, Hardt et al. 2016). Learns
  group-specific thresholds that satisfy equalized odds. It needs the
  sensitive attribute *at decision time*, which would mean using gender to
  decide a loan. It is computed as an analysis benchmark of what an
  equalized-odds-constrained policy would cost in accuracy, not deployed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.base import BaseEstimator, ClassifierMixin

from credit_risk.responsible_ai.fairness import MISSING_GROUP, group_labels


def reweighing_weights(
    y: np.ndarray | pd.Series, sensitive: pd.DataFrame, min_cell_count: int = 30
) -> np.ndarray:
    """Weights w(g, y) = P(g) * P(y) / P(g, y) over the intersection of groups.

    Intersections smaller than ``min_cell_count`` keep weight 1.0: a handful of
    rows would otherwise receive extreme weights and dominate the fit.
    The weights are normalized to mean 1 so learning-rate semantics are unchanged.
    """
    labels = pd.Series(np.asarray(y, dtype=int), index=sensitive.index, name="_y")
    group = sensitive.astype("string").fillna(MISSING_GROUP).agg("|".join, axis=1).rename("_g")
    frame = pd.concat([group, labels], axis=1)
    n = len(frame)
    p_group = frame["_g"].map(frame["_g"].value_counts() / n)
    p_label = frame["_y"].map(frame["_y"].value_counts() / n)
    cell_counts = frame.groupby(["_g", "_y"])["_y"].transform("size")
    weights = (p_group * p_label) / (cell_counts / n)
    weights = weights.where(cell_counts >= min_cell_count, 1.0)
    return np.asarray(weights / weights.mean(), dtype=float)


class _ScorePassthrough(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """Treats the single input column as P(default) so thresholds act on the score."""

    def fit(self, x: np.ndarray, y: np.ndarray | pd.Series) -> _ScorePassthrough:
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        score = np.asarray(x, dtype=float).reshape(-1)
        return np.column_stack([1.0 - score, score])

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


@dataclass(frozen=True)
class ThresholdBenchmark:
    """Decisions from an equalized-odds ThresholdOptimizer, for comparison only."""

    attribute: str
    y_pred: np.ndarray


def threshold_optimizer_benchmark(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray,
    sensitive: pd.Series,
    seed: int = 42,
) -> ThresholdBenchmark:
    """Fit group-specific thresholds on already-computed scores.

    Uses ``prefit=True`` with an identity estimator over the score so the
    benchmark isolates the effect of the thresholds from the model itself.
    Fit on the evaluation set, so it is an optimistic upper bound on what
    post-processing could achieve, which is the right framing for a benchmark.
    """
    features = np.asarray(y_score, dtype=float).reshape(-1, 1)
    optimizer = ThresholdOptimizer(
        estimator=_ScorePassthrough().fit(features, y_true),
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
        predict_method="predict_proba",
    )
    groups = group_labels(sensitive)
    optimizer.fit(features, np.asarray(y_true, dtype=int), sensitive_features=groups)
    y_pred = optimizer.predict(features, sensitive_features=groups, random_state=seed)
    return ThresholdBenchmark(attribute=str(sensitive.name), y_pred=np.asarray(y_pred, dtype=int))
