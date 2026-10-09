import pytest

from ledgerflow_api.domain.fields import (
    FieldValueError,
    assess_fields,
    consistency_issues,
    normalize_field,
    overall_confidence,
)

VALID_NFE_KEY = "35261011222333000181550010000000011000000010"


@pytest.mark.parametrize(
    ("key", "raw", "expected"),
    [
        ("supplier_name", "  ACME Ltda  ", "ACME Ltda"),
        ("supplier_tax_id", "11.222.333/0001-81", "11222333000181"),
        ("supplier_tax_id", "de123456789", "DE123456789"),
        ("issue_date", "2026-10-09", "2026-10-09"),
        ("due_date", None, None),
        ("currency", "BRL", "BRL"),
        ("total_amount", "1530.4", "1530.40"),
        ("nfe_access_key", VALID_NFE_KEY, VALID_NFE_KEY),
        (
            "line_items",
            [{"description": "Widget", "quantity": "2", "unit_price": "10.5", "total": "21"}],
            [{"description": "Widget", "quantity": "2", "unit_price": "10.5", "total": "21.00"}],
        ),
    ],
)
def test_normalize_field_accepts(key: str, raw: object, expected: object) -> None:
    assert normalize_field(key, raw) == expected


@pytest.mark.parametrize(
    ("key", "raw"),
    [
        ("supplier_name", ""),
        ("supplier_tax_id", "11.222.333/0001-80"),
        ("issue_date", "09/10/2026"),
        ("issue_date", "20261009"),
        ("issue_date", "2026-02-30"),
        ("issue_date", None),
        ("currency", "real"),
        ("total_amount", 1530.4),
        ("nfe_access_key", "1234"),
        ("line_items", [{"description": "Widget", "quantity": "1"}]),
        ("unknown_field", "x"),
    ],
)
def test_normalize_field_rejects(key: str, raw: object) -> None:
    with pytest.raises(FieldValueError) as excinfo:
        normalize_field(key, raw)
    assert excinfo.value.key == key


def test_consistency_rules() -> None:
    issues = consistency_issues(
        {
            "issue_date": "2026-10-09",
            "due_date": "2026-10-01",
            "total_amount": "100.00",
            "tax_amount": "150.00",
            "line_items": [
                {"description": "A", "quantity": "2", "unit_price": "10", "total": "25.00"}
            ],
        },
        "invoice",
    )
    assert {issue.key for issue in issues} == {"due_date", "tax_amount", "line_items"}


def test_nfe_requires_matching_issuer() -> None:
    issues = consistency_issues(
        {"supplier_tax_id": "45997418000153", "nfe_access_key": VALID_NFE_KEY}, "nfe"
    )
    assert [issue.key for issue in issues] == ["nfe_access_key"]
    assert (
        consistency_issues(
            {"supplier_tax_id": "11222333000181", "nfe_access_key": VALID_NFE_KEY}, "nfe"
        )
        == []
    )


def test_assess_fields_keeps_raw_value_of_invalid_fields() -> None:
    assessment = assess_fields(
        {"supplier_tax_id": "11.222.333/0001-80", "total_amount": "10.5"}, "invoice"
    )
    assert assessment.values == {"supplier_tax_id": "11.222.333/0001-80", "total_amount": "10.50"}
    assert [issue.key for issue in assessment.issues] == ["supplier_tax_id"]


def test_overall_confidence_is_the_weakest_field() -> None:
    assert overall_confidence([0.99, 0.6, 0.9]) == 0.6
    assert overall_confidence([]) == 0.0
