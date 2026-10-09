"""The extracted field set: per-field normalization and cross-field consistency rules.

The same rules serve two callers with different strictness. Human corrections are rejected
outright when invalid; worker output is accepted as-is but every violation is reported so the
document is routed to human review.
"""

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from ledgerflow_api.domain.money import format_decimal, parse_decimal, parse_money
from ledgerflow_api.domain.tax_ids import (
    is_valid_cnpj,
    is_valid_nfe_access_key,
    nfe_issuer_cnpj,
    normalize_cnpj,
)

DocType = Literal["invoice", "nfe"]
FieldSource = Literal["llm", "rule", "human"]


class FieldKey(StrEnum):
    SUPPLIER_NAME = "supplier_name"
    SUPPLIER_TAX_ID = "supplier_tax_id"
    INVOICE_NUMBER = "invoice_number"
    ISSUE_DATE = "issue_date"
    DUE_DATE = "due_date"
    CURRENCY = "currency"
    TOTAL_AMOUNT = "total_amount"
    TAX_AMOUNT = "tax_amount"
    NFE_ACCESS_KEY = "nfe_access_key"
    LINE_ITEMS = "line_items"


NULLABLE_FIELDS = frozenset({FieldKey.DUE_DATE, FieldKey.NFE_ACCESS_KEY})

type JsonValue = str | int | float | bool | list[JsonValue] | dict[str, JsonValue] | None


class FieldValueError(ValueError):
    def __init__(self, key: str, message: str) -> None:
        super().__init__(f"{key}: {message}")
        self.key = key
        self.message = message


@dataclass(frozen=True, slots=True)
class FieldIssue:
    key: str
    message: str


_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_CURRENCY = re.compile(r"^[A-Z]{3}$")
_FOREIGN_TAX_ID = re.compile(r"^[A-Z0-9]{4,20}$")
_LINE_ITEM_KEYS = frozenset({"description", "quantity", "unit_price", "total"})
# NF-e allows up to 4 decimal places for quantities and 10 for unit prices.
_QUANTITY_PLACES = 4
_UNIT_PRICE_PLACES = 10
_LINE_TOTAL_TOLERANCE = Decimal("0.01")


def _text(max_length: int) -> Callable[[object], str]:
    def normalize(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("must be a non-empty string")
        text = value.strip()
        if len(text) > max_length:
            raise ValueError(f"must be at most {max_length} characters")
        return text

    return normalize


def _iso_date(value: object) -> str:
    if not isinstance(value, str) or not _ISO_DATE.fullmatch(value):
        raise ValueError("must be an ISO date (YYYY-MM-DD)")
    return date.fromisoformat(value).isoformat()


def _tax_id(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("must be a string")
    candidate = normalize_cnpj(value).upper()
    if len(candidate) == 14 and candidate.isdigit():
        if not is_valid_cnpj(candidate):
            raise ValueError("CNPJ check digits do not match")
        return candidate
    if not _FOREIGN_TAX_ID.fullmatch(candidate):
        raise ValueError("must be a CNPJ or a 4-20 character alphanumeric VAT id")
    return candidate


def _currency(value: object) -> str:
    if not isinstance(value, str) or not _CURRENCY.fullmatch(value):
        raise ValueError("must be an ISO 4217 code such as 'BRL'")
    return value


def _money(value: object) -> str:
    return format_decimal(parse_money(value))


def _access_key(value: object) -> str:
    if not isinstance(value, str) or not is_valid_nfe_access_key(value):
        raise ValueError("must be 44 digits with a valid mod-11 check digit")
    return value.replace(" ", "")


def _line_items(value: object) -> list[JsonValue]:
    if not isinstance(value, list):
        raise ValueError("must be a list")
    items: list[JsonValue] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != _LINE_ITEM_KEYS:
            raise ValueError(f"item {index} must have exactly {sorted(_LINE_ITEM_KEYS)}")
        try:
            items.append(
                {
                    "description": _text(500)(item["description"]),
                    "quantity": format_decimal(
                        parse_decimal(item["quantity"], max_places=_QUANTITY_PLACES)
                    ),
                    "unit_price": format_decimal(
                        parse_decimal(item["unit_price"], max_places=_UNIT_PRICE_PLACES)
                    ),
                    "total": format_decimal(parse_money(item["total"])),
                }
            )
        except ValueError as exc:
            raise ValueError(f"item {index}: {exc}") from exc
    return items


_NORMALIZERS: Mapping[FieldKey, Callable[[object], JsonValue]] = {
    FieldKey.SUPPLIER_NAME: _text(255),
    FieldKey.SUPPLIER_TAX_ID: _tax_id,
    FieldKey.INVOICE_NUMBER: _text(64),
    FieldKey.ISSUE_DATE: _iso_date,
    FieldKey.DUE_DATE: _iso_date,
    FieldKey.CURRENCY: _currency,
    FieldKey.TOTAL_AMOUNT: _money,
    FieldKey.TAX_AMOUNT: _money,
    FieldKey.NFE_ACCESS_KEY: _access_key,
    FieldKey.LINE_ITEMS: _line_items,
}


def parse_field_key(key: str) -> FieldKey:
    try:
        return FieldKey(key)
    except ValueError:
        raise FieldValueError(key, "unknown field") from None


def normalize_field(key: str, value: object) -> JsonValue:
    """Return the canonical form of ``value`` or raise ``FieldValueError``."""
    field = parse_field_key(key)
    if value is None:
        if field in NULLABLE_FIELDS:
            return None
        raise FieldValueError(key, "must not be null")
    try:
        return _NORMALIZERS[field](value)
    except ValueError as exc:
        raise FieldValueError(key, str(exc)) from exc


def consistency_issues(
    values: Mapping[str, JsonValue], doc_type: DocType | None
) -> list[FieldIssue]:
    """Cross-field rules over already-normalized values."""
    issues: list[FieldIssue] = []
    issue_date, due_date = values.get(FieldKey.ISSUE_DATE), values.get(FieldKey.DUE_DATE)
    if isinstance(issue_date, str) and isinstance(due_date, str) and due_date < issue_date:
        issues.append(FieldIssue(FieldKey.DUE_DATE, "is before issue_date"))

    total, tax = values.get(FieldKey.TOTAL_AMOUNT), values.get(FieldKey.TAX_AMOUNT)
    if isinstance(total, str) and isinstance(tax, str) and Decimal(tax) > Decimal(total):
        issues.append(FieldIssue(FieldKey.TAX_AMOUNT, "exceeds total_amount"))

    line_items = values.get(FieldKey.LINE_ITEMS)
    if isinstance(line_items, list):
        for index, item in enumerate(line_items):
            if not isinstance(item, dict):
                continue
            expected = Decimal(str(item["quantity"])) * Decimal(str(item["unit_price"]))
            if abs(expected - Decimal(str(item["total"]))) > _LINE_TOTAL_TOLERANCE:
                issues.append(
                    FieldIssue(FieldKey.LINE_ITEMS, f"item {index}: quantity * unit_price != total")
                )

    tax_id, access_key = values.get(FieldKey.SUPPLIER_TAX_ID), values.get(FieldKey.NFE_ACCESS_KEY)
    if doc_type == "nfe":
        if access_key is None:
            issues.append(FieldIssue(FieldKey.NFE_ACCESS_KEY, "is required for NF-e documents"))
        if isinstance(tax_id, str) and not is_valid_cnpj(tax_id):
            issues.append(FieldIssue(FieldKey.SUPPLIER_TAX_ID, "must be a CNPJ for NF-e documents"))
    if (
        isinstance(access_key, str)
        and isinstance(tax_id, str)
        and is_valid_cnpj(tax_id)
        and nfe_issuer_cnpj(access_key) != tax_id
    ):
        issues.append(
            FieldIssue(FieldKey.NFE_ACCESS_KEY, "issuer CNPJ does not match supplier_tax_id")
        )
    return issues


@dataclass(frozen=True, slots=True)
class FieldAssessment:
    """Result of checking worker output: canonical values where possible, plus every issue."""

    values: dict[str, JsonValue]
    issues: list[FieldIssue]


def assess_fields(raw: Mapping[str, JsonValue], doc_type: DocType | None) -> FieldAssessment:
    values: dict[str, JsonValue] = {}
    issues: list[FieldIssue] = []
    for key, value in raw.items():
        try:
            values[key] = normalize_field(key, value)
        except FieldValueError as exc:
            # Keep the raw value so reviewers see exactly what the model produced.
            values[key] = value
            issues.append(FieldIssue(exc.key, exc.message))
    valid = {k: v for k, v in values.items() if k not in {i.key for i in issues}}
    issues.extend(consistency_issues(valid, doc_type))
    return FieldAssessment(values=values, issues=issues)


def overall_confidence(confidences: Iterable[float]) -> float:
    """The weakest field decides: a document is only as trustworthy as its least certain value."""
    return min(confidences, default=0.0)
