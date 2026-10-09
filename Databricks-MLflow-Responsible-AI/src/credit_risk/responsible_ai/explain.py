"""SHAP explanations: global importance and per-application reason codes.

Global importance documents what drives the model (model card, reviewers).
Local explanations produce the top contributing factors for an individual
decision, the raw material for adverse-action style notices that tell a
declined producer which factors weighed against them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_risk.config import CATEGORICAL_FEATURES
from credit_risk.training.model import feature_matrix


@dataclass(frozen=True)
class Explanations:
    global_importance: pd.DataFrame  # columns: feature, mean_abs_shap
    local: list[dict[str, Any]]  # one entry per explained application


def _positive_class_values(raw: Any) -> np.ndarray:
    # SHAP's return shape for binary classifiers differs across versions and
    # model types: a list per class, a (n, f, 2) array, or a (n, f) array.
    if isinstance(raw, list):
        return np.asarray(raw[-1])
    values = np.asarray(raw)
    return values[:, :, -1] if values.ndim == 3 else values


def _source_feature(transformed_name: str) -> str:
    """Fold one-hot columns (e.g. ``crop_type_soy``) back to their source feature."""
    for column in CATEGORICAL_FEATURES:
        if transformed_name.startswith(f"{column}_"):
            return column
    return transformed_name


def explain(
    pipeline: Pipeline,
    df: pd.DataFrame,
    ids: pd.Series,
    *,
    n_local: int = 20,
    top_k: int = 5,
) -> Explanations:
    """Explain the fitted pipeline on ``df``.

    Local explanations are produced for the ``n_local`` highest-risk rows, the
    applications most likely to be declined and therefore most in need of a
    reason. Contributions are in log-odds space, where they are additive.
    """
    import shap

    transformed = pipeline[:-1].transform(feature_matrix(df))
    explainer = shap.TreeExplainer(pipeline[-1])
    values = _positive_class_values(explainer.shap_values(transformed))
    names = [_source_feature(str(c)) for c in transformed.columns]

    per_source = pd.DataFrame(values, columns=names).T.groupby(level=0).sum().T
    importance = (
        per_source.abs()
        .mean()
        .sort_values(ascending=False)
        .rename("mean_abs_shap")
        .rename_axis("feature")
        .reset_index()
    )

    scores = pipeline[-1].predict_proba(transformed)[:, 1]
    local: list[dict[str, Any]] = []
    for pos in np.argsort(-scores)[:n_local]:
        contributions = per_source.iloc[pos].sort_values(key=np.abs, ascending=False).head(top_k)
        local.append(
            {
                "loan_id": str(ids.iloc[pos]),
                "predicted_default_probability": float(scores[pos]),
                "top_factors": [
                    {
                        "feature": feature,
                        "value": _display_value(df, feature, pos),
                        "shap_log_odds": float(contribution),
                        "direction": "increases risk" if contribution > 0 else "decreases risk",
                    }
                    for feature, contribution in contributions.items()
                ],
            }
        )
    return Explanations(global_importance=importance, local=local)


def _display_value(df: pd.DataFrame, feature: str, pos: int) -> Any:
    value = df[feature].iloc[pos]
    if isinstance(value, np.generic):
        return value.item()
    return None if pd.isna(value) else value
