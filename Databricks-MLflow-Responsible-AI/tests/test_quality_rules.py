from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.synthetic import GeneratorConfig, generate_loans
from credit_risk.pipeline import prepare_silver
from credit_risk.quality.cleaning import clean_applications, deduplicate_latest, farm_size_band
from credit_risk.quality.rules import (
    DataQualityError,
    InRange,
    InSet,
    NoRescuedData,
    NotNull,
    annotate_violations,
    split_valid,
)
from credit_risk.schemas import DQ_FAILED_RULES_COLUMN


def test_not_null() -> None:
    df = pd.DataFrame({"a": ["x", None, np.nan]})
    assert NotNull("a").violations(df).tolist() == [False, True, True]


@pytest.mark.parametrize(
    ("rule", "values", "expected"),
    [
        (InRange("x", 0, 10), [0, 10, -1, 11, None], [False, False, True, True, True]),
        (InRange("x", 0, 10, exclusive_min=True), [0, 0.1], [True, False]),
        (InRange("x", 0, 10, allow_null=True), [None, 5], [False, False]),
    ],
)
def test_in_range(rule: InRange, values: list[float | None], expected: list[bool]) -> None:
    df = pd.DataFrame({"x": pd.array(values, dtype="Float64")})
    assert rule.violations(df).tolist() == expected


def test_in_range_treats_non_numeric_as_violation() -> None:
    df = pd.DataFrame({"x": ["5", "five"]})
    assert InRange("x", 0, 10).violations(df).tolist() == [False, True]


def test_in_set_with_nullable_ints() -> None:
    df = pd.DataFrame({"y": pd.array([0, 1, 2, None], dtype="Int64")})
    assert InSet("y", frozenset({0, 1}), allow_null=True).violations(df).tolist() == [
        False,
        False,
        True,
        False,
    ]
    assert InSet("y", frozenset({0, 1})).violations(df).tolist() == [False, False, True, True]


def test_no_rescued_data_is_noop_without_the_column() -> None:
    assert not NoRescuedData().violations(pd.DataFrame({"a": [1]})).any()
    df = pd.DataFrame({"_rescued_data": [None, '{"loan_amount_brl":"abc"}']})
    assert NoRescuedData().violations(df).tolist() == [False, True]


def test_annotate_lists_every_broken_rule() -> None:
    df = pd.DataFrame({"a": [1, None], "x": [5, 50]})
    annotated = annotate_violations(df, rules=(NotNull("a"), InRange("x", 0, 10)))
    assert annotated[DQ_FAILED_RULES_COLUMN].tolist() == [[], ["a_not_null", "x_in_range"]]


def test_annotate_handles_empty_frames() -> None:
    annotated = annotate_violations(pd.DataFrame({"a": []}), rules=(NotNull("a"),))
    assert DQ_FAILED_RULES_COLUMN in annotated.columns


def test_clean_generated_data_passes_every_rule() -> None:
    raw = generate_loans(GeneratorConfig(n_loans=3_000, corrupt_fraction=0.0, seed=2))
    result = prepare_silver(raw)
    assert result.quarantined.empty
    assert len(result.valid) == len(raw)


def test_injected_defects_are_quarantined_with_reasons(raw_loans: pd.DataFrame) -> None:
    result = prepare_silver(raw_loans)
    counts = result.violation_counts()
    # Half of the 2% corrupted rows carry unfixable defects, cycling over four kinds.
    assert len(result.quarantined) == 60
    assert set(counts) == {
        "loan_amount_brl_in_range",
        "producer_id_not_null",
        "crop_type_in_set",
        "interest_rate_in_range",
    }
    # The other half only had formatting noise, which cleaning repairs.
    assert (
        result.valid["region"].isin(["Sul", "Sudeste", "Centro-Oeste", "Nordeste", "Norte"]).all()
    )


def test_batch_tolerance_stops_the_pipeline(raw_loans: pd.DataFrame) -> None:
    annotated = annotate_violations(clean_applications(raw_loans))
    with pytest.raises(DataQualityError, match="exceeds tolerance"):
        split_valid(annotated, max_quarantine_rate=0.001)


def test_cleaning_normalizes_codes_without_guessing() -> None:
    df = pd.DataFrame(
        {
            "region": ["  NORDESTE ", "sul", "Atlantis"],
            "gender": ["feminino", " M ", "x"],
            "crop_type": [" Soy", "CORN", "coffee"],
            "state": [" ba", "rs", "sp"],
            "farm_size_ha": [10.0, 60.0, 600.0],
        }
    )
    cleaned = clean_applications(df)
    assert cleaned["region"].tolist() == ["Nordeste", "Sul", "Atlantis"]
    assert cleaned["gender"].tolist() == ["F", "M", "x"]
    assert cleaned["crop_type"].tolist() == ["soy", "corn", "coffee"]
    assert cleaned["state"].tolist() == ["BA", "RS", "SP"]


def test_farm_size_band_edges_are_lower_inclusive() -> None:
    bands = farm_size_band(pd.Series([0.0, 0.1, 49.9, 50.0, 499.9, 500.0, None, -3.0]))
    assert bands.tolist() == [
        pd.NA,
        "smallholder",
        "smallholder",
        "medium",
        "medium",
        "large",
        pd.NA,
        pd.NA,
    ]


def test_deduplicate_keeps_latest_ingested_record() -> None:
    df = pd.DataFrame(
        {
            "loan_id": ["A", "A", "B"],
            "loan_amount_brl": [100.0, 120.0, 50.0],
            "_ingested_at": pd.to_datetime(["2025-01-02", "2025-01-01", "2025-01-01"]),
        }
    )
    out = deduplicate_latest(df).set_index("loan_id")
    assert out.loc["A", "loan_amount_brl"] == 100.0
    assert len(out) == 2
