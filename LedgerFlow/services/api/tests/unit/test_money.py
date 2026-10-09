from decimal import Decimal

import pytest

from ledgerflow_api.domain.money import (
    DecimalFormatError,
    format_decimal,
    micros_to_usd,
    parse_decimal,
    parse_money,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("1530.40", "1530.40"), ("1530.4", "1530.40"), ("0", "0.00"), (" 12.5 ", "12.50")],
)
def test_parse_money_pads_without_rounding(raw: str, expected: str) -> None:
    assert format_decimal(parse_money(raw)) == expected


@pytest.mark.parametrize(
    "raw",
    [
        1530.40,  # floats are refused outright, even when they look exact
        1530,
        "1530.401",  # more precision than cents: reject, never round
        "1,530.40",
        "1e3",
        "NaN",
        "Infinity",
        "-5.00",
        "",
        None,
    ],
)
def test_parse_money_rejects(raw: object) -> None:
    with pytest.raises(DecimalFormatError):
        parse_money(raw)


def test_parse_decimal_respects_place_limit_and_sign() -> None:
    assert parse_decimal("3.1416", max_places=4) == Decimal("3.1416")
    assert parse_decimal("-2.5", max_places=1, allow_negative=True) == Decimal("-2.5")
    with pytest.raises(DecimalFormatError):
        parse_decimal("3.14159", max_places=4)


def test_format_decimal_never_uses_exponent_notation() -> None:
    assert format_decimal(Decimal("1E+2")) == "100"
    assert format_decimal(Decimal("0.0000001")) == "0.0000001"


@pytest.mark.parametrize(
    ("micros", "expected"), [(0, "0.000000"), (412, "0.000412"), (12_345_678, "12.345678")]
)
def test_micros_to_usd(micros: int, expected: str) -> None:
    assert micros_to_usd(micros) == expected
