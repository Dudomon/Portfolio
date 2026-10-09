"""Brazilian fiscal identifiers: CNPJ and the NF-e access key (chave de acesso)."""

import re

_CNPJ_FIRST_WEIGHTS = (5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2)
_CNPJ_SECOND_WEIGHTS = (6, *_CNPJ_FIRST_WEIGHTS)
_CNPJ_FORMATTING = re.compile(r"[.\-/\s]")

NFE_ACCESS_KEY_LENGTH = 44
# Positions 7-20 of the access key hold the issuer's CNPJ (Manual de Orientação do Contribuinte).
NFE_ISSUER_CNPJ_SLICE = slice(6, 20)


def _mod11_check_digit(digits: str, weights: tuple[int, ...]) -> int:
    remainder = sum(int(d) * w for d, w in zip(digits, weights, strict=True)) % 11
    return 0 if remainder < 2 else 11 - remainder


def normalize_cnpj(raw: str) -> str:
    """Strip the usual ``00.000.000/0000-00`` punctuation."""
    return _CNPJ_FORMATTING.sub("", raw)


def is_valid_cnpj(raw: str) -> bool:
    cnpj = normalize_cnpj(raw)
    if len(cnpj) != 14 or not cnpj.isascii() or not cnpj.isdigit():
        return False
    # Repeated digits pass the checksum arithmetic but are never issued.
    if len(set(cnpj)) == 1:
        return False
    first = _mod11_check_digit(cnpj[:12], _CNPJ_FIRST_WEIGHTS)
    second = _mod11_check_digit(cnpj[:12] + str(first), _CNPJ_SECOND_WEIGHTS)
    return cnpj[12:] == f"{first}{second}"


def nfe_check_digit(first_43: str) -> int:
    """Mod-11 check digit with weights 2..9 applied right-to-left; remainders 0/1 map to 0."""
    total = sum(int(d) * (2 + i % 8) for i, d in enumerate(reversed(first_43)))
    remainder = total % 11
    return 0 if remainder < 2 else 11 - remainder


def is_valid_nfe_access_key(raw: str) -> bool:
    key = raw.replace(" ", "")
    if len(key) != NFE_ACCESS_KEY_LENGTH or not key.isascii() or not key.isdigit():
        return False
    return int(key[-1]) == nfe_check_digit(key[:-1])


def nfe_issuer_cnpj(access_key: str) -> str:
    return access_key.replace(" ", "")[NFE_ISSUER_CNPJ_SLICE]
