from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.evaluation import evaluate
from credit_risk.responsible_ai.calibration import (
    calibration_by_group,
    expected_calibration_error,
    reliability_table,
)
from credit_risk.responsible_ai.fairness import MISSING_GROUP, assess_fairness, group_labels
from credit_risk.responsible_ai.mitigation import (
    reweighing_weights,
    threshold_optimizer_benchmark,
)


@pytest.fixture
def toy() -> tuple[np.ndarray, np.ndarray, pd.Series]:
    # Group A: 4 rows, y = [1,1,0,0], pred = [1,0,1,0] -> sel 0.50, TPR 0.5, FPR 0.5
    # Group B: 4 rows, y = [1,1,0,0], pred = [1,1,0,0] -> sel 0.50, TPR 1.0, FPR 0.0
    # Group C: 4 rows, y = [1,0,0,0], pred = [0,0,0,0] -> sel 0.00, TPR 0.0, FPR 0.0
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0])
    pred = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    groups = pd.Series(list("AAAABBBBCCCC"), name="region")
    return y, pred, groups


def test_metric_frame_by_group(toy: tuple[np.ndarray, np.ndarray, pd.Series]) -> None:
    summary = assess_fairness(*toy)
    by_group = summary.by_group
    assert by_group.loc["A", "selection_rate"] == 0.5
    assert by_group.loc["B", "tpr"] == 1.0
    assert by_group.loc["A", "fpr"] == 0.5
    assert by_group.loc["C", "count"] == 4
    assert summary.min_group_size == 4


def test_between_group_differences(toy: tuple[np.ndarray, np.ndarray, pd.Series]) -> None:
    summary = assess_fairness(*toy)
    assert summary.demographic_parity_difference == pytest.approx(0.5)
    assert summary.false_positive_rate_difference == pytest.approx(0.5)
    # Equalized odds takes the larger of the TPR gap (1.0) and FPR gap (0.5).
    assert summary.equalized_odds_difference == pytest.approx(1.0)


def test_identical_treatment_has_zero_gaps() -> None:
    y = np.array([1, 0, 1, 0])
    summary = assess_fairness(y, y, pd.Series(["F", "F", "M", "M"], name="gender"))
    assert summary.demographic_parity_difference == 0
    assert summary.equalized_odds_difference == 0


def test_missing_attribute_values_form_their_own_group() -> None:
    labels = group_labels(pd.Series(["F", None, "M"], dtype="string"))
    assert labels.tolist() == ["F", MISSING_GROUP, "M"]


def test_reweighing_makes_label_independent_of_group() -> None:
    rng = np.random.default_rng(0)
    group = pd.DataFrame({"g": rng.choice(["a", "b"], size=4_000, p=[0.7, 0.3])})
    # Base rates differ strongly: 10% in a, 30% in b.
    y = (rng.random(4_000) < np.where(group["g"] == "a", 0.1, 0.3)).astype(int)
    weights = reweighing_weights(y, group)
    assert weights.mean() == pytest.approx(1.0)
    frame = group.assign(y=y, w=weights)
    weighted_rate = frame.groupby("g").apply(
        lambda part: np.average(part["y"], weights=part["w"]), include_groups=False
    )
    assert weighted_rate["a"] == pytest.approx(weighted_rate["b"], abs=1e-9)


def test_reweighing_leaves_tiny_cells_alone() -> None:
    group = pd.DataFrame({"g": ["a"] * 100 + ["b"] * 5})
    y = np.array([0, 1] * 50 + [1] * 5)
    weights = reweighing_weights(y, group, min_cell_count=30)
    # Cell (a, 0): P(a) P(0) / P(a, 0) = (100/105)(50/105) / (50/105) = 100/105.
    # Cell (b, 1) has 5 rows, below the minimum, so it keeps raw weight 1.0.
    # Mean normalization scales both equally, so their ratio is preserved.
    assert weights[-1] / weights[0] == pytest.approx(105 / 100)
    assert np.allclose(weights[-5:], weights[-1])


def test_threshold_optimizer_benchmark_returns_binary_decisions() -> None:
    rng = np.random.default_rng(1)
    n = 2_000
    groups = pd.Series(rng.choice(["x", "y"], size=n), name="region")
    score = np.clip(rng.normal(np.where(groups == "x", 0.3, 0.5), 0.15), 0, 1)
    y = (rng.random(n) < score).astype(int)
    benchmark = threshold_optimizer_benchmark(y, score, groups)
    assert benchmark.attribute == "region"
    assert set(np.unique(benchmark.y_pred)) <= {0, 1}
    assert benchmark.y_pred.shape == (n,)


def test_ece_is_zero_when_predictions_match_bin_rates() -> None:
    y = np.array([0] * 9 + [1] * 1 + [0] * 5 + [1] * 5)
    p = np.array([0.1] * 10 + [0.5] * 10)
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_ece_detects_overconfidence() -> None:
    y = np.array([0] * 9 + [1])
    p = np.full(10, 0.9)
    assert expected_calibration_error(y, p) == pytest.approx(0.8)


def test_reliability_table_counts_probability_one() -> None:
    table = reliability_table(np.array([1, 0]), np.array([1.0, 0.0]), n_bins=10)
    assert table["count"].sum() == 2
    assert table["bin_upper"].max() == 1.0


def test_calibration_by_group_reports_each_group() -> None:
    table = calibration_by_group(
        np.array([0, 1, 0, 1]), np.array([0.2, 0.8, 0.4, 0.6]), pd.Series(["a", "a", "b", "b"])
    )
    assert set(table.index) == {"a", "b"}
    assert table.loc["a", "observed_rate"] == 0.5


def test_evaluate_flattens_metrics_for_mlflow() -> None:
    rng = np.random.default_rng(2)
    score = rng.random(400)
    y = (rng.random(400) < score).astype(int)
    sensitive = pd.DataFrame({"gender": rng.choice(["F", "M"], 400)})
    report = evaluate(y, score, sensitive, threshold=0.5)
    metrics = report.flat_metrics("holdout_")
    assert {"holdout_roc_auc", "holdout_gender_dp_diff", "holdout_gender_fpr_diff"} <= set(metrics)
    assert report.performance.flag_rate == pytest.approx((score >= 0.5).mean())
