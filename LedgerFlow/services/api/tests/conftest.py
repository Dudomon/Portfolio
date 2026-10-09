from typing import Any

from ledgerflow_api.config import Settings

TEST_JWT_SECRET = "test-only-jwt-secret-with-enough-entropy"
TEST_SERVICE_TOKEN = "test-only-service-token"
UNUSED_DATABASE_URL = "postgresql+asyncpg://unused:unused@127.0.0.1:1/unused"


def make_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "environment": "test",
        "database_url": UNUSED_DATABASE_URL,
        "jwt_secret": TEST_JWT_SECRET,
        "internal_service_token": TEST_SERVICE_TOKEN,
        "log_json": False,
        "log_level": "WARNING",
        "s3_endpoint_url": "http://storage.invalid",
        "s3_access_key_id": "test",
        "s3_secret_access_key": "test",
    }
    return Settings(_env_file=None, **(values | overrides))
