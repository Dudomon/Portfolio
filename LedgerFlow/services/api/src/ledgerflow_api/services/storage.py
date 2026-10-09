"""Object storage for uploaded documents (S3 or any S3-compatible store such as MinIO)."""

import uuid
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol, Self
from urllib.parse import quote

import aioboto3
from aiobotocore.config import AioConfig

from ledgerflow_api.config import Settings

if TYPE_CHECKING:
    from types_aiobotocore_s3 import S3Client


class ObjectStorage(Protocol):
    async def put(self, key: str, data: bytes, *, content_type: str) -> None: ...

    async def presigned_get_url(self, key: str, *, filename: str, content_type: str) -> str: ...


def storage_key(tenant_id: uuid.UUID, document_id: uuid.UUID, sha256: str) -> str:
    return f"tenants/{tenant_id}/documents/{document_id}/{sha256}"


def content_disposition(filename: str) -> str:
    # RFC 6266/5987 encoding: never interpolate a user-supplied name into a quoted header value.
    return f"inline; filename*=UTF-8''{quote(filename, safe='')}"


class S3Storage:
    def __init__(self, client: "S3Client", presigner: "S3Client", settings: Settings) -> None:
        self._client = client
        self._presigner = presigner
        self._bucket = settings.s3_bucket
        self._sse = settings.s3_server_side_encryption
        self._url_ttl = settings.presigned_url_ttl_seconds

    @classmethod
    @asynccontextmanager
    async def connect(cls, settings: Settings) -> AsyncIterator[Self]:
        session = aioboto3.Session()
        credentials: dict[str, Any] = {"region_name": settings.s3_region}
        if settings.s3_access_key_id and settings.s3_secret_access_key:
            credentials["aws_access_key_id"] = settings.s3_access_key_id.get_secret_value()
            credentials["aws_secret_access_key"] = settings.s3_secret_access_key.get_secret_value()
        # S3-compatible stores generally need path-style addressing (no bucket subdomains).
        addressing = "path" if settings.s3_endpoint_url else "auto"
        config = AioConfig(signature_version="s3v4", s3={"addressing_style": addressing})

        async with AsyncExitStack() as stack:
            client = await stack.enter_async_context(
                session.client(
                    "s3", endpoint_url=settings.s3_endpoint_url, config=config, **credentials
                )
            )
            presigner = client
            if settings.s3_public_endpoint_url:
                # The signature covers the host, so URLs for browsers must be signed against the
                # public endpoint rather than rewritten afterwards.
                presigner = await stack.enter_async_context(
                    session.client(
                        "s3",
                        endpoint_url=settings.s3_public_endpoint_url,
                        config=config,
                        **credentials,
                    )
                )
            yield cls(client, presigner, settings)

    async def put(self, key: str, data: bytes, *, content_type: str) -> None:
        extra: dict[str, Any] = {}
        if self._sse:
            extra["ServerSideEncryption"] = self._sse
        await self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType=content_type, **extra
        )

    async def presigned_get_url(self, key: str, *, filename: str, content_type: str) -> str:
        return await self._presigner.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self._bucket,
                "Key": key,
                "ResponseContentType": content_type,
                "ResponseContentDisposition": content_disposition(filename),
            },
            ExpiresIn=self._url_ttl,
        )
