from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIB = 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = True

    database_url: str = Field(description="SQLAlchemy URL, e.g. postgresql+asyncpg://...")
    db_pool_size: int = 10
    db_max_overflow: int = 5

    jwt_algorithm: Literal["HS256", "RS256"] = "HS256"
    jwt_secret: SecretStr | None = None
    jwt_private_key: SecretStr | None = None
    jwt_public_key: str | None = None
    jwt_jwks_url: str | None = None
    access_token_ttl_seconds: int = 900

    internal_service_token: SecretStr

    s3_bucket: str = "ledgerflow-documents"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str | None = None
    # Presigned URLs are opened by browsers, which may not resolve the in-cluster endpoint.
    s3_public_endpoint_url: str | None = None
    s3_access_key_id: SecretStr | None = None
    s3_secret_access_key: SecretStr | None = None
    s3_server_side_encryption: Literal["AES256", "aws:kms"] | None = "AES256"
    presigned_url_ttl_seconds: int = 300

    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    events_exchange: str = "ledgerflow.events"
    outbox_batch_size: int = Field(default=100, ge=1, le=1000)
    outbox_poll_interval_seconds: float = Field(default=0.5, gt=0)

    max_upload_bytes: int = 15 * MIB

    cors_origins: list[str] = Field(default_factory=list)
    otel_exporter_otlp_endpoint: str | None = None
    otel_service_name: str = "ledgerflow-api"

    @model_validator(mode="after")
    def _check_jwt_keys(self) -> Self:
        if self.jwt_algorithm == "HS256":
            if self.jwt_secret is None or len(self.jwt_secret.get_secret_value()) < 32:
                raise ValueError("HS256 requires JWT_SECRET of at least 32 characters")
        else:
            if self.jwt_private_key is None:
                raise ValueError("RS256 requires JWT_PRIVATE_KEY to issue tokens")
            if self.jwt_public_key is None and self.jwt_jwks_url is None:
                raise ValueError("RS256 requires JWT_PUBLIC_KEY or JWT_JWKS_URL to verify tokens")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
