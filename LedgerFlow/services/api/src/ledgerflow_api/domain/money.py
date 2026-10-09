"""Decimal handling for monetary values.

Amounts travel as strings and are never routed through ``float``: binary floating point cannot
represent most cents exactly, and silently rounding someone's invoice total is not acceptable.
Values with more precision than allowed are rejected instead of rounded.
"""

import re
from decimal import Decimal

_DECIMAL_PATTERN = re.compile(r"^-?\d{1,15}(?:\.\d+)?$")

MONEY_PLACES = 2


class DecimalFormatError(ValueError):
    pass


def parse_decimal(raw: object, *, max_places: int, allow_negative: bool = False) -> Decimal:
    """Parse a plain decimal string such as ``"1530.40"``.

    Rejects floats, exponent notation, NaN/Infinity and thousands separators so the only
    accepted representation is the canonical one the contract specifies.
    """
    if not isinstance(raw, str):
        raise DecimalFormatError(f"expected a decimal string, got {type(raw).__name__}")
    text = raw.strip()
    if not _DECIMAL_PATTERN.fullmatch(text):
        raise DecimalFormatError(f"'{raw}' is not a plain decimal number")
    _, _, fraction = text.partition(".")
    if len(fraction) > max_places:
        raise DecimalFormatError(f"'{raw}' has more than {max_places} decimal places")
    value = Decimal(text)
    if value < 0 and not allow_negative:
        raise DecimalFormatError(f"'{raw}' must not be negative")
    return value


def parse_money(raw: object) -> Decimal:
    return quantize(parse_decimal(raw, max_places=MONEY_PLACES), MONEY_PLACES)


def quantize(value: Decimal, places: int) -> Decimal:
    """Pad to a fixed number of places. Only ever widens, so no rounding mode is involved."""
    return value.quantize(Decimal(1).scaleb(-places))


def format_decimal(value: Decimal) -> str:
    """Render without exponent notation (``str(Decimal("1E+2"))`` would give ``"1E+2"``)."""
    return format(value, "f")


def micros_to_usd(micros: int) -> str:
    return format_decimal(quantize(Decimal(micros).scaleb(-6), 6))
