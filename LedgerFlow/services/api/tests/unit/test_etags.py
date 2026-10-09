import pytest

from ledgerflow_api.errors import PreconditionFailedError, PreconditionRequiredError
from ledgerflow_api.etags import check_if_match, etag_for


def test_etag_format() -> None:
    assert etag_for(7) == 'W/"7"'


@pytest.mark.parametrize("header", ['W/"7"', '"7"', '"3", W/"7"', "*"])
def test_matching_headers_pass(header: str) -> None:
    check_if_match(header, 7)


@pytest.mark.parametrize("header", [None, "", "   "])
def test_missing_header_requires_precondition(header: str | None) -> None:
    with pytest.raises(PreconditionRequiredError):
        check_if_match(header, 7)


@pytest.mark.parametrize("header", ['W/"6"', "7", '"77"', 'W/"6", "8"'])
def test_stale_or_malformed_header_fails(header: str) -> None:
    with pytest.raises(PreconditionFailedError) as excinfo:
        check_if_match(header, 7)
    assert excinfo.value.extensions == {"current_etag": 'W/"7"'}
