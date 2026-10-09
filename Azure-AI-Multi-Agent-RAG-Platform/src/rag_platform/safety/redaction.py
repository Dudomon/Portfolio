"""PII redaction applied to log records before they leave the process.

Questions about procurement and payroll routinely contain employee emails, phone
numbers, bank details or tax identifiers. Telemetry retention and access are broader
than the application's, so values are masked at the logging boundary rather than
trusting every call site to remember.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from re import Match

_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){2,7}(?:\s?[A-Z0-9]{1,4})?\b")
_US_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CARD_CANDIDATE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")
# A separator after the area code is required so bare document numbers
# (purchase orders, invoices) are not mistaken for phone numbers.
_PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[\s.-]?)?\(?\d{2,4}\)?[\s.-]\d{3,4}[\s.-]?\d{3,4}(?!\w)")


def _luhn_valid(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        value = int(char)
        if position % 2 == 1:
            value *= 2
            if value > 9:  # noqa: PLR2004 - Luhn digit folding
                value -= 9
        total += value
    return total % 10 == 0


def _mask_card(match: Match[str]) -> str:
    digits = re.sub(r"\D", "", match.group(0))
    # Only mask strings that pass the Luhn check, so PO and invoice numbers survive.
    return "[REDACTED_CARD]" if _luhn_valid(digits) else match.group(0)


_RULES: tuple[tuple[re.Pattern[str], str | Callable[[Match[str]], str]], ...] = (
    (_EMAIL, "[REDACTED_EMAIL]"),
    (_IBAN, "[REDACTED_IBAN]"),
    (_US_SSN, "[REDACTED_SSN]"),
    (_CARD_CANDIDATE, _mask_card),
    (_PHONE, "[REDACTED_PHONE]"),
)


def redact(text: str) -> str:
    """Mask common PII patterns. Order matters: card numbers are checked before phones."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text
