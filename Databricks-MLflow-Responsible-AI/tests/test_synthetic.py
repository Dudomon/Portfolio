"""The generator must contain exactly the biases it documents, and nothing else."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_risk.data.synthetic import BiasConfig, GeneratorConfig, generate_loans
from credit_risk.schemas import BRONZE_COLUMNS


def test_columns_match_bronze_schema(raw_loans: pd.DataFrame) -> None:
    assert list(raw_loans.columns) == [name for name, _ in BRONZE_COLUMNS]


def test_same_seed_is_reproducible() -> None:
    config = GeneratorConfig(n_loans=500, seed=3)
    pd.testing.assert_frame_equal(generate_loans(config), generate_loans(config))


def test_different_seeds_differ() -> None:
    a = generate_loans(GeneratorConfig(n_loans=500, seed=1))
    b = generate_loans(GeneratorConfig(n_loans=500, seed=2))
    assert not a["loan_amount_brl"].equals(b["loan_amount_brl"])


def test_unlabeled_batch_has_null_labels() -> None:
    df = generate_loans(GeneratorConfig(n_loans=200), include_label=False)
    assert df["defaulted"].isna().all()


def test_gender_measurement_bias_hides_history_but_not_risk(latent_loans: pd.DataFrame) -> None:
    by_gender = latent_loans.groupby("gender")
    visible_share = (
        by_gender["credit_history_months"].sum() / by_gender["_true_credit_history_months"].sum()
    )
    # Women's history is under-recorded by the configured visibility factor...
    assert visible_share["F"] == pytest.approx(BiasConfig().female_history_visibility, abs=0.01)
    assert visible_share["M"] == pytest.approx(1.0)
    # ...while their true default risk is the same as men's.
    true_risk = by_gender["_true_default_prob"].mean()
    assert abs(true_risk["F"] - true_risk["M"]) < 0.01


def test_nordeste_label_bias_inflates_observed_defaults(latent_loans: pd.DataFrame) -> None:
    by_region = latent_loans.groupby("region")
    excess = by_region["defaulted"].mean() - by_region["_true_default_prob"].mean()
    assert excess["Nordeste"] > 0.04
    assert excess.drop("Nordeste").abs().max() < 0.02


def test_unbiased_config_removes_injected_biases() -> None:
    df = generate_loans(
        GeneratorConfig(n_loans=20_000, corrupt_fraction=0.0, bias=BiasConfig.unbiased(), seed=7),
        include_latent=True,
    )
    assert (df["credit_history_months"] == df["_true_credit_history_months"]).all()
    nordeste = df[df["region"] == "Nordeste"]
    assert abs(nordeste["defaulted"].mean() - nordeste["_true_default_prob"].mean()) < 0.02


def test_smallholders_have_less_insurance_by_design(latent_loans: pd.DataFrame) -> None:
    small = latent_loans["farm_size_ha"] < 50
    insured = latent_loans["has_crop_insurance"]
    assert insured[small].mean() < insured[~small].mean() - 0.2


def test_corruption_rate_is_respected() -> None:
    clean = generate_loans(GeneratorConfig(n_loans=2_000, corrupt_fraction=0.0, seed=5))
    dirty = generate_loans(GeneratorConfig(n_loans=2_000, corrupt_fraction=0.05, seed=5))
    changed = (clean.astype(str) != dirty.astype(str)).any(axis=1)
    assert changed.sum() == 100
