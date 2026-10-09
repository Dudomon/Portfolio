from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from credit_risk.data.synthetic import GeneratorConfig, generate_loans
from credit_risk.governance.policy import PromotionPolicy, load_policy

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def policy() -> PromotionPolicy:
    return load_policy(PROJECT_ROOT / "conf" / "promotion_policy.yml")


@pytest.fixture(scope="session")
def latent_loans() -> pd.DataFrame:
    """Biased synthetic data with latent ground-truth columns, no corruption."""
    return generate_loans(
        GeneratorConfig(n_loans=20_000, corrupt_fraction=0.0, seed=7), include_latent=True
    )


@pytest.fixture(scope="session")
def raw_loans() -> pd.DataFrame:
    """Bronze-shaped extract including deliberately corrupted rows."""
    return generate_loans(GeneratorConfig(n_loans=6_000, corrupt_fraction=0.02, seed=11))
