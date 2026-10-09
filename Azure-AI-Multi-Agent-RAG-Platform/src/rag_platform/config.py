"""Application settings loaded from environment variables (and `.env` for local runs)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelPrice(BaseModel):
    """Price per one million tokens, in USD, for a single deployment."""

    input_per_million: float = Field(ge=0)
    output_per_million: float = Field(ge=0)


def _default_prices() -> dict[str, ModelPrice]:
    # Illustrative list prices keyed by deployment name. Real prices depend on the
    # region, the deployment type and the enterprise agreement, so they are meant
    # to be overridden through MODEL_PRICES (JSON) per environment.
    return {
        "gpt-4o": ModelPrice(input_per_million=2.50, output_per_million=10.00),
        "gpt-4o-mini": ModelPrice(input_per_million=0.15, output_per_million=0.60),
        "text-embedding-3-large": ModelPrice(input_per_million=0.13, output_per_million=0.0),
    }


class Settings(BaseSettings):
    """Runtime configuration. Every Azure resource name is configurable per environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", frozen=True)

    # Azure OpenAI. When the API key is empty the service authenticates with
    # DefaultAzureCredential, which resolves to the managed identity in Azure.
    azure_openai_endpoint: str = "https://example.openai.azure.com"
    azure_openai_api_key: SecretStr | None = None
    azure_openai_api_version: str = "2024-10-21"
    chat_deployment: str = "gpt-4o"
    # Planning, query rewriting and verification are classification-like tasks,
    # so they run on a cheaper model by default.
    utility_deployment: str = "gpt-4o-mini"
    judge_deployment: str = "gpt-4o"
    embedding_deployment: str = "text-embedding-3-large"
    embedding_dimensions: int = Field(default=1536, gt=0)

    # Azure AI Search.
    azure_search_endpoint: str = "https://example.search.windows.net"
    azure_search_api_key: SecretStr | None = None
    search_index_name: str = "policies-v1"
    semantic_configuration_name: str = "policies-semantic"

    # Retrieval tuning.
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    vector_k_nearest_neighbors: int = Field(default=50, ge=1)
    min_reranker_score: float = Field(default=1.2, ge=0, le=4)
    max_evidence_chunks: int = Field(default=8, ge=1)
    max_sub_questions: int = Field(default=3, ge=1, le=5)

    # Guardrails for a single request.
    request_token_budget: int = Field(default=24_000, gt=0)
    request_timeout_seconds: float = Field(default=60.0, gt=0)
    step_timeout_seconds: float = Field(default=25.0, gt=0)
    max_revisions: int = Field(default=1, ge=0, le=3)
    groundedness_threshold: float = Field(default=1.0, ge=0, le=1)

    # Telemetry.
    applicationinsights_connection_string: SecretStr | None = None
    service_name: str = "rag-platform"
    log_level: str = "INFO"

    model_prices: dict[str, ModelPrice] = Field(default_factory=_default_prices)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
