import pytest

from ledgerflow_api.domain.tax_ids import (
    is_valid_cnpj,
    is_valid_nfe_access_key,
    nfe_check_digit,
    nfe_issuer_cnpj,
)

# Weights for the 43 leading digits, written out left to right (2..9 cycling from the right),
# as an independent statement of the algorithm under test.
NFE_WEIGHTS = "4329876543298765432987654329876543298765432"
VALID_NFE_KEY = "35261011222333000181550010000000011000000010"


def _reference_nfe_digit(first_43: str) -> int:
    remainder = sum(int(d) * int(w) for d, w in zip(first_43, NFE_WEIGHTS, strict=True)) % 11
    return 0 if remainder < 2 else 11 - remainder


@pytest.mark.parametrize(
    "cnpj", ["11.222.333/0001-81", "11222333000181", "11 222 333 0001 81", "45.997.418/0001-53"]
)
def test_valid_cnpj(cnpj: str) -> None:
    assert is_valid_cnpj(cnpj)


@pytest.mark.parametrize(
    "cnpj",
    [
        "11.222.333/0001-82",  # wrong second digit
        "11.222.333/0001-91",  # wrong first digit
        "00000000000000",  # passes the arithmetic, never issued
        "1122233300018",  # too short
        "112223330001811",  # too long
        "1122233300018a",
        "",
    ],
)
def test_invalid_cnpj(cnpj: str) -> None:
    assert not is_valid_cnpj(cnpj)


def test_nfe_check_digit_matches_reference() -> None:
    first_43 = VALID_NFE_KEY[:-1]
    assert nfe_check_digit(first_43) == _reference_nfe_digit(first_43) == int(VALID_NFE_KEY[-1])


def test_valid_nfe_key() -> None:
    assert is_valid_nfe_access_key(VALID_NFE_KEY)
    assert is_valid_nfe_access_key(" ".join(VALID_NFE_KEY[i : i + 4] for i in range(0, 44, 4)))


@pytest.mark.parametrize(
    "key",
    [
        VALID_NFE_KEY[:-1] + str((int(VALID_NFE_KEY[-1]) + 1) % 10),
        VALID_NFE_KEY[:-1],
        VALID_NFE_KEY + "0",
        "A" + VALID_NFE_KEY[1:],
    ],
)
def test_invalid_nfe_key(key: str) -> None:
    assert not is_valid_nfe_access_key(key)


def test_issuer_cnpj_is_embedded_in_key() -> None:
    assert nfe_issuer_cnpj(VALID_NFE_KEY) == "11222333000181"
