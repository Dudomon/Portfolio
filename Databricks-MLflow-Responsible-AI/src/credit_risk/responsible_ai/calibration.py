"""Calibration checks.

A credit score is used as a probability (pricing, provisioning, thresholds),
so it must be calibrated overall *and* within each group; a model can have
equal error rates across groups while systematically over-stating risk for one
of them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from credit_risk.responsible_ai.fairness import group_labels


def reliability_table(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, n_bins: int = 10
) -> pd.DataFrame:
    """Equal-width bins of predicted probability vs observed default rate."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Right-closed last bin so p == 1.0 is counted.
    bins = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    frame = pd.DataFrame({"bin": bins, "y": y, "p": p})
    table = frame.groupby("bin").agg(
        count=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean")
    )
    table["bin_lower"] = edges[table.index.to_numpy()]
    table["bin_upper"] = edges[table.index.to_numpy() + 1]
    return table.reset_index(drop=True)


def expected_calibration_error(
    y_true: np.ndarray | pd.Series, y_prob: np.ndarray | pd.Series, n_bins: int = 10
) -> float:
    """Count-weighted mean |observed - predicted| across bins."""
    table = reliability_table(y_true, y_prob, n_bins)
    if table.empty:
        return 0.0
    gaps = (table["observed_rate"] - table["mean_predicted"]).abs()
    return float((gaps * table["count"]).sum() / table["count"].sum())


def calibration_by_group(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    groups: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Brier score, ECE, and mean predicted vs observed rate for each group."""
    frame = pd.DataFrame(
        {
            "y": np.asarray(y_true, dtype=float),
            "p": np.asarray(y_prob, dtype=float),
            "group": group_labels(groups),
        }
    )
    rows = []
    for group, part in frame.groupby("group"):
        rows.append(
            {
                "group": group,
                "count": len(part),
                "mean_predicted": float(part["p"].mean()),
                "observed_rate": float(part["y"].mean()),
                "brier": float(brier_score_loss(part["y"], part["p"])),
                "ece": expected_calibration_error(part["y"], part["p"], n_bins),
            }
        )
    return pd.DataFrame(rows).set_index("group")
