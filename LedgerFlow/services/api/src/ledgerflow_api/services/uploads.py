"""Upload intake: bounded reading, hashing and content sniffing.

The client-supplied ``Content-Type`` is ignored. The stored type is derived from the file's
leading bytes, so a renamed executable can't be served back to reviewers as a "PDF".
"""

import hashlib
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Protocol

from ledgerflow_api.errors import (
    PayloadTooLargeError,
    UnprocessableError,
    UnsupportedMediaTypeError,
)

_CHUNK_SIZE = 64 * 1024
_UTF8_BOM = b"\xef\xbb\xbf"
_MAX_FILENAME_LENGTH = 255

_BINARY_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "application/pdf"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
)
# An XML declaration or one of the NF-e root elements; a bare "<" would also admit HTML.
_XML_PREFIXES = (b"<?xml", b"<nfeProc", b"<NFe")


class AsyncReadable(Protocol):
    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True, slots=True)
class UploadedFile:
    data: bytes
    sha256: str
    content_type: str
    filename: str

    @property
    def size_bytes(self) -> int:
        return len(self.data)


def sniff_content_type(head: bytes) -> str | None:
    for signature, content_type in _BINARY_SIGNATURES:
        if head.startswith(signature):
            return content_type
    text_head = head.removeprefix(_UTF8_BOM).lstrip()
    if text_head.startswith(_XML_PREFIXES):
        return "application/xml"
    return None


def safe_filename(raw: str | None) -> str:
    """Keep only the final path component; browsers and some clients send full local paths."""
    name = PurePosixPath((raw or "").replace("\\", "/")).name.strip()
    return name[:_MAX_FILENAME_LENGTH] or "upload"


async def read_upload(
    source: AsyncReadable, *, filename: str | None, max_bytes: int
) -> UploadedFile:
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    size = 0
    while chunk := await source.read(_CHUNK_SIZE):
        size += len(chunk)
        if size > max_bytes:
            raise PayloadTooLargeError(f"Files are limited to {max_bytes} bytes.")
        digest.update(chunk)
        chunks.append(chunk)
    if size == 0:
        raise UnprocessableError("The uploaded file is empty.")
    data = b"".join(chunks)
    content_type = sniff_content_type(data[:512])
    if content_type is None:
        raise UnsupportedMediaTypeError("Only PDF, PNG, JPEG and XML files are accepted.")
    return UploadedFile(
        data=data,
        sha256=digest.hexdigest(),
        content_type=content_type,
        filename=safe_filename(filename),
    )
