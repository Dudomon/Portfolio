"""Fictitious crop-financing loans for Brazilian agricultural producers.

Every number in this module is invented. The generator exists so the fairness
and governance tooling has something real to detect, which means bias is
injected on purpose and with known mechanics (see :class:`BiasConfig`):

* **Measurement bias (gender).** Many women farmers operate land whose credit
  history is recorded under a spouse's tax ID. The *observed*
  ``credit_history_months`` for women is therefore a fraction of their true
  history, while default risk depends on the true history. A model trained on
  the observed column over-estimates risk for women even though gender is never
  a feature.
* **Label bias (region).** Drought-driven renegotiations in the Nordeste are
  recorded as defaults in the historical book. The *observed* label carries an
  extra logit shift for that region that the true risk does not.
* **Structural correlation (farm size).** Smallholders are less likely to hold
  crop insurance, which genuinely raises their risk. This is not injected bias;
  it is there so the documentation can discuss the difference between unfair
  bias and real risk that correlates with a protected group.

Representation imbalance (about one in five holders is a woman, Norte is a small
region) is realistic and makes per-group estimates noisier for minority groups.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from credit_risk.domain import CROPS, REGIONS

REGION_WEIGHTS: dict[str, float] = {
    "Sul": 0.30,
    "Sudeste": 0.22,
    "Centro-Oeste": 0.20,
    "Nordeste": 0.20,
    "Norte": 0.08,
}

REGION_STATES: dict[str, tuple[str, ...]] = {
    "Sul": ("RS", "PR", "SC"),
    "Sudeste": ("MG", "SP", "ES"),
    "Centro-Oeste": ("MT", "GO", "MS"),
    "Nordeste": ("BA", "PE", "CE", "MA", "PI"),
    "Norte": ("PA", "TO", "RO"),
}

# Crop mix per region, in the order of CROPS.
REGION_CROP_MIX: dict[str, tuple[float, ...]] = {
    "Sul": (0.40, 0.30, 0.02, 0.03, 0.05, 0.20),
    "Sudeste": (0.15, 0.20, 0.30, 0.25, 0.02, 0.08),
    "Centro-Oeste": (0.55, 0.25, 0.01, 0.07, 0.02, 0.10),
    "Nordeste": (0.15, 0.20, 0.05, 0.15, 0.30, 0.15),
    "Norte": (0.25, 0.15, 0.05, 0.02, 0.23, 0.30),
}

# Median farm size (ha) per region; spread is log-normal.
REGION_MEDIAN_FARM_HA: dict[str, float] = {
    "Sul": 45.0,
    "Sudeste": 60.0,
    "Centro-Oeste": 400.0,
    "Nordeste": 18.0,
    "Norte": 90.0,
}

# Gross revenue per hectare (BRL) by crop, before noise.
CROP_REVENUE_PER_HA: dict[str, float] = {
    "soy": 6500.0,
    "corn": 5200.0,
    "coffee": 14000.0,
    "sugarcane": 9000.0,
    "cassava": 4200.0,
    "beef_cattle": 1800.0,
}

# Mean rainfall anomaly (standardized) by region; the Nordeste is drought-prone.
REGION_RAINFALL_MEAN: dict[str, float] = {
    "Sul": 0.0,
    "Sudeste": 0.0,
    "Centro-Oeste": -0.1,
    "Nordeste": -0.6,
    "Norte": 0.2,
}

TERMS_MONTHS: tuple[int, ...] = (12, 24, 36, 60)

if set(REGION_WEIGHTS) != set(REGIONS) or set(CROP_REVENUE_PER_HA) != set(CROPS):
    raise RuntimeError("Generator tables are out of sync with credit_risk.domain")


@dataclass(frozen=True)
class BiasConfig:
    """Knobs for the injected biases. ``BiasConfig.unbiased()`` turns them off."""

    # Fraction of a woman's true credit history that is visible to the bureau.
    female_history_visibility: float = 0.35
    # Extra logit applied to *observed* defaults in the Nordeste.
    nordeste_label_shift: float = 0.7

    @classmethod
    def unbiased(cls) -> BiasConfig:
        return cls(female_history_visibility=1.0, nordeste_label_shift=0.0)


@dataclass(frozen=True)
class GeneratorConfig:
    n_loans: int = 20_000
    loans_per_producer: float = 2.0
    female_share: float = 0.2
    start_date: date = date(2022, 1, 1)
    span_days: int = 3 * 365
    # Share of rows deliberately corrupted so data-quality rules have work to do.
    corrupt_fraction: float = 0.01
    bias: BiasConfig = field(default_factory=BiasConfig)
    seed: int = 42


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_loans(
    config: GeneratorConfig | None = None,
    *,
    include_label: bool = True,
    include_latent: bool = False,
) -> pd.DataFrame:
    """Generate a raw loan-application extract shaped like the bronze table.

    Args:
        config: Generator settings; defaults to :class:`GeneratorConfig`.
        include_label: When False the ``defaulted`` column is null, mimicking
            new applications that still need scoring.
        include_latent: Adds ``_true_default_prob`` and
            ``_true_credit_history_months`` so tests can verify that the
            injected bias behaves as documented. Never written to the lakehouse.
    """
    cfg = config or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_loans
    n_producers = max(1, int(n / cfg.loans_per_producer))

    # --- Producer-level attributes (stable across a producer's loans) -------
    regions = np.array(list(REGION_WEIGHTS))
    p_region = np.array(list(REGION_WEIGHTS.values()))
    prod_region = rng.choice(regions, size=n_producers, p=p_region / p_region.sum())
    prod_state = np.array([rng.choice(REGION_STATES[r]) for r in prod_region])
    prod_gender = np.where(rng.random(n_producers) < cfg.female_share, "F", "M")
    median_ha = np.array([REGION_MEDIAN_FARM_HA[r] for r in prod_region])
    # Floor at 0.5 ha so rounding never produces a zero-hectare farm.
    prod_farm_ha = np.maximum(np.round(median_ha * rng.lognormal(0.0, 1.0, n_producers), 1), 0.5)
    prod_years = rng.integers(1, 45, n_producers)
    # True formal credit history at the start of the observation window.
    prod_history_start = np.clip(
        prod_years * 12 * rng.uniform(0.3, 1.0, n_producers), 0, 480
    ).astype(int)
    prod_coop = rng.random(n_producers) < np.where(prod_region == "Sul", 0.65, 0.35)
    prod_irrigated = rng.random(n_producers) < np.where(prod_farm_ha > 200, 0.35, 0.12)
    prod_prior_defaults = rng.poisson(0.25, n_producers)

    # --- Loan-level attributes ---------------------------------------------
    owner = rng.integers(0, n_producers, n)
    region = prod_region[owner]
    gender = prod_gender[owner]
    farm_ha = prod_farm_ha[owner]
    day_offsets = rng.integers(0, cfg.span_days, n)
    application_date = pd.to_datetime(cfg.start_date) + pd.to_timedelta(day_offsets, unit="D")
    months_elapsed = day_offsets // 30

    crop_idx = np.array([rng.choice(len(CROPS), p=REGION_CROP_MIX[r]) for r in region])
    crop = np.array(CROPS)[crop_idx]
    revenue_per_ha = np.array([CROP_REVENUE_PER_HA[c] for c in crop])
    annual_revenue = np.round(farm_ha * revenue_per_ha * rng.lognormal(0.0, 0.35, n), 2)
    loan_amount = np.round(annual_revenue * rng.lognormal(np.log(0.45), 0.5, n), 2)
    existing_debt = np.round(annual_revenue * rng.gamma(1.5, 0.25, n), 2)
    term = rng.choice(TERMS_MONTHS, size=n, p=(0.45, 0.25, 0.15, 0.15))
    # Smallholders access subsidized programs (e.g. family-farming credit lines).
    base_rate = np.where(farm_ha < 50, 0.05, 0.09)
    interest_rate = np.round(base_rate + rng.normal(0.0, 0.015, n).clip(-0.03, 0.06), 4)
    insurance_p = np.where(farm_ha < 50, 0.25, np.where(farm_ha < 500, 0.55, 0.75))
    has_insurance = rng.random(n) < insurance_p
    rain_mean = np.array([REGION_RAINFALL_MEAN[r] for r in region])
    rainfall_anomaly = np.round(rng.normal(rain_mean, 1.0), 3)

    true_history = prod_history_start[owner] + months_elapsed
    visibility = np.where(gender == "F", cfg.bias.female_history_visibility, 1.0)
    observed_history = np.floor(true_history * visibility).astype(int)

    # --- Default mechanism ---------------------------------------------------
    loan_to_revenue = loan_amount / annual_revenue
    debt_to_revenue = existing_debt / annual_revenue
    logit = (
        -1.9
        + 1.4 * np.clip(loan_to_revenue, 0, 3)
        + 0.6 * np.clip(debt_to_revenue, 0, 3)
        + 0.55 * prod_prior_defaults[owner]
        - 0.8 * np.log1p(true_history / 24.0)
        - 0.6 * has_insurance
        - 0.4 * prod_irrigated[owner]
        - 0.3 * prod_coop[owner]
        - 0.55 * rainfall_anomaly
        + 6.0 * (interest_rate - 0.07)
    )
    true_p = _sigmoid(logit)
    observed_logit = logit + np.where(region == "Nordeste", cfg.bias.nordeste_label_shift, 0.0)
    defaulted = (rng.random(n) < _sigmoid(observed_logit)).astype(int)

    df = pd.DataFrame(
        {
            "loan_id": [f"L{i:08d}" for i in range(n)],
            "producer_id": [f"P{o:07d}" for o in owner],
            "application_date": application_date.date,
            "region": region,
            "state": prod_state[owner],
            "gender": gender,
            "farm_size_ha": farm_ha,
            "crop_type": crop,
            "loan_amount_brl": loan_amount,
            "term_months": term.astype(int),
            "interest_rate": interest_rate,
            "annual_revenue_brl": annual_revenue,
            "existing_debt_brl": existing_debt,
            "years_farming": prod_years[owner].astype(int) + (months_elapsed // 12),
            "credit_history_months": observed_history,
            "prior_defaults": prod_prior_defaults[owner].astype(int),
            "has_crop_insurance": has_insurance,
            "irrigated": prod_irrigated[owner],
            "cooperative_member": prod_coop[owner],
            "rainfall_anomaly": rainfall_anomaly,
            "defaulted": pd.array(defaulted if include_label else [pd.NA] * n, dtype="Int64"),
        }
    )
    if include_latent:
        df["_true_default_prob"] = true_p
        df["_true_credit_history_months"] = true_history
    return _corrupt(df, cfg.corrupt_fraction, rng)


def _corrupt(df: pd.DataFrame, fraction: float, rng: np.random.Generator) -> pd.DataFrame:
    """Inject the kinds of defects seen in real extracts.

    Half of the corrupted rows get recoverable formatting noise (whitespace,
    casing) that silver cleaning must fix; the other half get defects that the
    data-quality rules must quarantine.
    """
    n_bad = int(len(df) * fraction)
    if n_bad == 0:
        return df
    out = df.copy()
    rows = rng.choice(len(out), size=n_bad, replace=False)
    fixable, broken = np.array_split(rows, 2)
    out["region"] = out["region"].astype(object)
    out.loc[out.index[fixable], "region"] = [f"  {r.upper()} " for r in out["region"].iloc[fixable]]
    defects = ("negative_amount", "missing_producer", "unknown_crop", "bad_rate")
    for i, row in enumerate(broken):
        idx = out.index[row]
        kind = defects[i % len(defects)]
        if kind == "negative_amount":
            out.loc[idx, "loan_amount_brl"] = -abs(float(out.loc[idx, "loan_amount_brl"]))
        elif kind == "missing_producer":
            out.loc[idx, "producer_id"] = None
        elif kind == "unknown_crop":
            out.loc[idx, "crop_type"] = "tobacco_unknown"
        else:
            out.loc[idx, "interest_rate"] = 4.5  # percent entered instead of fraction
    return out
