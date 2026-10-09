"""Group fairness metrics with fairlearn.

Convention: the positive prediction is "flagged as likely to default", which
in this use case means the application is declined or sent to manual review.
So:

* **selection rate** = share of the group that is flagged (declined);
* **FPR** = share of producers who *would have repaid* but were flagged: the
  main harm, i.e. creditworthy farmers denied credit;
* **TPR** = share of actual defaulters caught: the lender's protection.

Demographic parity difference compares selection rates; equalized odds
difference is the larger of the TPR and FPR gaps. docs/RESPONSIBLE_AI.md
explains why the FPR gap is the primary gate and why the TPR side of equalized
odds gets a looser threshold (few defaults per group make TPR noisy).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    count,
    demographic_parity_difference,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)

MISSING_GROUP = "<missing>"


def group_labels(sensitive: pd.Series) -> np.ndarray:
    """String group labels with nulls as an explicit group.

    Dropping rows with a missing attribute would hide exactly the records the
    data owner most needs to see, so they are reported as their own group.
    """
    return np.asarray(sensitive.astype("string").fillna(MISSING_GROUP).to_numpy(), dtype=object)


@dataclass(frozen=True)
class FairnessSummary:
    attribute: str
    demographic_parity_difference: float
    equalized_odds_difference: float
    by_group: pd.DataFrame  # index: group; columns: count, selection_rate, tpr, fpr

    @property
    def min_group_size(self) -> int:
        return int(self.by_group["count"].min())

    @property
    def false_positive_rate_difference(self) -> float:
        """Gap in the share of would-be repayers who get flagged: the direct harm."""
        fpr = self.by_group["fpr"]
        return float(fpr.max() - fpr.min())

    def to_dict(self) -> dict[str, object]:
        return {
            "attribute": self.attribute,
            "demographic_parity_difference": self.demographic_parity_difference,
            "equalized_odds_difference": self.equalized_odds_difference,
            "false_positive_rate_difference": self.false_positive_rate_difference,
            "by_group": {
                str(group): {k: float(v) for k, v in row.items()}
                for group, row in self.by_group.iterrows()
            },
        }


def assess_fairness(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive: pd.Series,
) -> FairnessSummary:
    """Per-group rates and between-group differences for one sensitive attribute."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    groups = group_labels(sensitive)
    frame = MetricFrame(
        metrics={
            "count": count,
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
        },
        y_true=y_true_arr,
        y_pred=y_pred_arr,
        sensitive_features=pd.Series(groups, name=sensitive.name),
    )
    return FairnessSummary(
        attribute=str(sensitive.name),
        demographic_parity_difference=float(
            demographic_parity_difference(y_true_arr, y_pred_arr, sensitive_features=groups)
        ),
        equalized_odds_difference=float(
            equalized_odds_difference(y_true_arr, y_pred_arr, sensitive_features=groups)
        ),
        by_group=frame.by_group.astype(float),
    )


def assess_all(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    sensitive_frame: pd.DataFrame,
) -> dict[str, FairnessSummary]:
    return {
        str(column): assess_fairness(y_true, y_pred, sensitive_frame[column])
        for column in sensitive_frame.columns
    }
