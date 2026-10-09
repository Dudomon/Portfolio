from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.config import MODEL_FEATURES, NUMERIC_FEATURES, SENSITIVE_ATTRIBUTES
from credit_risk.features.transforms import (
    annual_installment,
    join_point_in_time,
    loan_features,
    producer_features,
    safe_ratio,
)
from credit_risk.pipeline import build_model_frame, prepare_silver
from credit_risk.training.model import feature_matrix, temporal_split


def test_sensitive_attributes_are_never_model_features() -> None:
    assert not set(SENSITIVE_ATTRIBUTES) & set(MODEL_FEATURES)


def test_safe_ratio_returns_nan_for_non_positive_denominators() -> None:
    out = safe_ratio(pd.Series([1.0, 1.0, 1.0]), pd.Series([2.0, 0.0, -1.0]))
    assert out.iloc[0] == 0.5
    assert out.iloc[1:].isna().all()


def test_annual_installment_matches_price_table() -> None:
    # 12,000 at 12% a year over 12 months: monthly payment 1,066.19.
    out = annual_installment(pd.Series([12_000.0]), pd.Series([0.12]), pd.Series([12]))
    assert out.iloc[0] == pytest.approx(1_066.1855 * 12, rel=1e-6)


def test_annual_installment_zero_rate_is_straight_line() -> None:
    out = annual_installment(pd.Series([1_200.0]), pd.Series([0.0]), pd.Series([24]))
    assert out.iloc[0] == pytest.approx(600.0)


def _applications() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "loan_id": ["L1", "L2", "L3"],
            "producer_id": ["P1", "P1", "P2"],
            "application_date": pd.to_datetime(["2024-01-10", "2024-06-01", "2024-03-01"]).date,
            "farm_size_ha": [100.0, 100.0, 20.0],
            "years_farming": [5, 6, 30],
            "credit_history_months": [24, 29, 300],
            "prior_defaults": [0, 1, 0],
            "cooperative_member": [True, True, False],
            "irrigated": [False, False, True],
            "annual_revenue_brl": [500_000.0, 400_000.0, 80_000.0],
            "existing_debt_brl": [100_000.0, 200_000.0, 0.0],
            "loan_amount_brl": [200_000.0, 100_000.0, 40_000.0],
            "interest_rate": [0.08, 0.09, 0.05],
            "term_months": [12, 24, 36],
            "has_crop_insurance": [True, False, False],
            "rainfall_anomaly": [0.1, -1.2, 0.0],
            "crop_type": ["soy", "soy", "cassava"],
            "defaulted": [0, 1, 0],
            "gender": ["M", "M", "F"],
        }
    )


def test_loan_features_carry_passthrough_columns() -> None:
    out = loan_features(_applications(), passthrough=("defaulted", "gender"))
    assert out["loan_to_revenue"].tolist() == pytest.approx([0.4, 0.25, 0.5])
    assert out["has_crop_insurance"].tolist() == [1, 0, 0]
    assert {"defaulted", "gender"} <= set(out.columns)


def test_producer_features_are_snapshotted_per_application_date() -> None:
    out = producer_features(_applications())
    assert len(out) == 3
    assert out.loc[0, "debt_to_revenue"] == pytest.approx(0.2)
    assert out.loc[1, "revenue_per_ha"] == pytest.approx(4_000.0)


def test_point_in_time_join_never_uses_future_snapshots() -> None:
    apps = _applications()
    loans = loan_features(apps)
    producers = producer_features(apps)
    joined = join_point_in_time(loans, producers).set_index("loan_id")
    # L1 (Jan) must see the January snapshot even though June's exists for P1.
    assert joined.loc["L1", "prior_defaults"] == 0
    assert joined.loc["L2", "prior_defaults"] == 1
    assert joined.loc["L3", "credit_history_months"] == 300


def test_point_in_time_join_leaves_gaps_as_missing() -> None:
    apps = _applications()
    early_loan = loan_features(
        apps.iloc[[0]].assign(application_date=[pd.Timestamp("2023-01-01").date()])
    )
    joined = join_point_in_time(early_loan, producer_features(apps))
    assert joined["farm_size_ha"].isna().all()


def test_model_frame_keeps_row_count_and_features(raw_loans: pd.DataFrame) -> None:
    silver = prepare_silver(raw_loans).valid
    frame = build_model_frame(silver)
    assert len(frame) == len(silver)
    matrix = feature_matrix(frame)
    assert list(matrix.columns) == list(MODEL_FEATURES)
    assert all(pd.api.types.is_float_dtype(t) for t in matrix[list(NUMERIC_FEATURES)].dtypes)
    assert matrix[list(NUMERIC_FEATURES)].notna().all().all()


def test_temporal_split_holds_out_the_most_recent_rows(raw_loans: pd.DataFrame) -> None:
    frame = build_model_frame(prepare_silver(raw_loans).valid)
    split = temporal_split(frame, holdout_fraction=0.25)
    train_dates = pd.to_datetime(split.train["application_date"])
    holdout_dates = pd.to_datetime(split.holdout["application_date"])
    assert train_dates.max() < pd.Timestamp(split.cutoff) <= holdout_dates.min()
    assert len(split.holdout) / len(frame) == pytest.approx(0.25, abs=0.01)


def test_temporal_split_rejects_bad_fraction(raw_loans: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="holdout_fraction"):
        temporal_split(raw_loans, holdout_fraction=1.0)
