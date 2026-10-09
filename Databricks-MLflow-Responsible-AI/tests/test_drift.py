from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from credit_risk.monitoring.drift import drift_level, psi_categorical, psi_numeric, psi_report


def test_identical_distributions_have_zero_psi() -> None:
    values = pd.Series(np.random.default_rng(0).normal(size=5_000))
    assert psi_numeric(values, values) == pytest.approx(0.0, abs=1e-12)


def test_independent_samples_of_same_distribution_are_stable() -> None:
    rng = np.random.default_rng(1)
    psi = psi_numeric(pd.Series(rng.normal(size=20_000)), pd.Series(rng.normal(size=20_000)))
    assert drift_level(psi) == "stable"


def test_mean_shift_is_significant() -> None:
    rng = np.random.default_rng(2)
    psi = psi_numeric(pd.Series(rng.normal(size=10_000)), pd.Series(rng.normal(1.0, size=10_000)))
    assert psi > 0.25
    assert drift_level(psi) == "significant"


def test_out_of_range_values_land_in_edge_bins() -> None:
    reference = pd.Series(np.linspace(0, 1, 1_000))
    current = pd.Series(np.full(1_000, 5.0))
    assert math.isfinite(psi_numeric(reference, current))
    assert psi_numeric(reference, current) > 1.0


def test_heavily_tied_counts_do_not_break_binning() -> None:
    reference = pd.Series([0] * 900 + [1] * 80 + [2] * 20)
    current = pd.Series([0] * 700 + [1] * 200 + [2] * 100)
    assert psi_numeric(reference, current) > 0.1


def test_categorical_psi_matches_hand_calculation() -> None:
    reference = pd.Series(["a"] * 50 + ["b"] * 50)
    current = pd.Series(["a"] * 80 + ["b"] * 20)
    expected = (0.8 - 0.5) * math.log(0.8 / 0.5) + (0.2 - 0.5) * math.log(0.2 / 0.5)
    assert psi_categorical(reference, current) == pytest.approx(expected)


def test_new_category_is_penalized_not_ignored() -> None:
    reference = pd.Series(["soy"] * 100)
    current = pd.Series(["soy"] * 90 + ["tobacco"] * 10)
    assert psi_categorical(reference, current) > 0.25


def test_report_covers_requested_columns() -> None:
    df = pd.DataFrame({"x": np.arange(100.0), "c": ["a", "b"] * 50})
    assert set(psi_report(df, df, numeric=["x"], categorical=["c"])) == {"x", "c"}


def test_empty_sample_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        psi_numeric(pd.Series([1.0, 2.0]), pd.Series([np.nan]))


@pytest.mark.parametrize(
    ("psi", "level"), [(0.0, "stable"), (0.099, "stable"), (0.1, "moderate"), (0.25, "significant")]
)
def test_drift_levels(psi: float, level: str) -> None:
    assert drift_level(psi) == level
