"""Population Stability Index (PSI).

PSI = sum((cur% - ref%) * ln(cur% / ref%)) over bins. It is the drift metric
credit-risk teams already use, so drift alerts are interpretable by the model
risk function without translation. Lakehouse Monitoring computes its own drift
statistics on the inference table; this in-package PSI is what the promotion
gate uses, because the gate must run before a model ever serves traffic.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd

DriftLevel = Literal["stable", "moderate", "significant"]

# Floor on bin proportions: an empty bin would make the log term infinite.
_EPSILON = 1e-4


def _psi_from_proportions(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(ref, _EPSILON, None)
    cur = np.clip(cur, _EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def psi_numeric(reference: pd.Series, current: pd.Series, n_bins: int = 10) -> float:
    """PSI with quantile bins fitted on the reference distribution.

    Duplicate quantile edges (common for counts such as ``prior_defaults``)
    are collapsed, and the outer edges are open so out-of-range current values
    land in the extreme bins instead of being dropped.
    """
    ref = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    cur = pd.to_numeric(current, errors="coerce").dropna().to_numpy(dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("PSI needs non-empty reference and current samples")
    inner_edges = np.unique(np.quantile(ref, np.linspace(0, 1, n_bins + 1)[1:-1]))
    ref_counts = np.bincount(
        np.searchsorted(inner_edges, ref, side="right"), minlength=inner_edges.size + 1
    )
    cur_counts = np.bincount(
        np.searchsorted(inner_edges, cur, side="right"), minlength=inner_edges.size + 1
    )
    return _psi_from_proportions(ref_counts / ref.size, cur_counts / cur.size)


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI over the union of observed categories (missing counted as a category)."""
    ref = reference.astype("string").fillna("<null>")
    cur = current.astype("string").fillna("<null>")
    categories = sorted(set(ref.unique()) | set(cur.unique()))
    ref_p = ref.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    cur_p = cur.value_counts(normalize=True).reindex(categories, fill_value=0.0).to_numpy()
    return _psi_from_proportions(ref_p, cur_p)


def psi_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: Iterable[str],
    categorical: Iterable[str] = (),
) -> dict[str, float]:
    report = {col: psi_numeric(reference[col], current[col]) for col in numeric}
    report.update({col: psi_categorical(reference[col], current[col]) for col in categorical})
    return report


def drift_level(psi: float) -> DriftLevel:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate"
    return "significant"
