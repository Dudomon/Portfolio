import hashlib

import pytest

from ledgerflow_api.errors import (
    PayloadTooLargeError,
    UnprocessableError,
    UnsupportedMediaTypeError,
)
from ledgerflow_api.services.uploads import read_upload, safe_filename, sniff_content_type


class ChunkedReader:
    """Mimics ``UploadFile.read`` while handing out deliberately small chunks."""

    def __init__(self, data: bytes, chunk: int = 7) -> None:
        self._data, self._chunk, self._pos = data, chunk, 0

    async def read(self, size: int = -1) -> bytes:
        step = min(size, self._chunk) if size > 0 else self._chunk
        out = self._data[self._pos : self._pos + step]
        self._pos += len(out)
        return out


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        (b"%PDF-1.7\n...", "application/pdf"),
        (b"\x89PNG\r\n\x1a\n\x00\x00", "image/png"),
        (b"\xff\xd8\xff\xe0JFIF", "image/jpeg"),
        (b'<?xml version="1.0"?><nfeProc/>', "application/xml"),
        (b"\xef\xbb\xbf\n  <?xml version='1.0'?>", "application/xml"),
        (b"<nfeProc versao='4.00'>", "application/xml"),
        (b"<html><body>hi</body></html>", None),
        (b"MZ\x90\x00", None),
        (b"plain text", None),
    ],
)
def test_sniff_content_type(head: bytes, expected: str | None) -> None:
    assert sniff_content_type(head) == expected


async def test_read_upload_hashes_and_sniffs() -> None:
    data = b"%PDF-1.7\n" + b"x" * 100
    upload = await read_upload(
        ChunkedReader(data), filename="C:\\Users\\me\\inv.pdf", max_bytes=1000
    )
    assert upload.sha256 == hashlib.sha256(data).hexdigest()
    assert upload.content_type == "application/pdf"
    assert upload.filename == "inv.pdf"
    assert upload.size_bytes == len(data)


async def test_read_upload_enforces_size_limit_while_streaming() -> None:
    with pytest.raises(PayloadTooLargeError):
        await read_upload(ChunkedReader(b"%PDF-" + b"x" * 50), filename="a.pdf", max_bytes=20)


async def test_read_upload_ignores_extension_and_rejects_unknown_content() -> None:
    with pytest.raises(UnsupportedMediaTypeError):
        await read_upload(ChunkedReader(b"#!/bin/sh\necho hi"), filename="a.pdf", max_bytes=100)


async def test_read_upload_rejects_empty_file() -> None:
    with pytest.raises(UnprocessableError):
        await read_upload(ChunkedReader(b""), filename="a.pdf", max_bytes=100)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd", "passwd"),
        ("dir/nota.xml", "nota.xml"),
        ("", "upload"),
        (None, "upload"),
    ],
)
def test_safe_filename(raw: str | None, expected: str) -> None:
    assert safe_filename(raw) == expected
