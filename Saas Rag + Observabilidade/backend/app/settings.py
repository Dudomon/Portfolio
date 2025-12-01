from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    app_name: str = "SaaS RAG Observability"
    environment: str = Field(default="local")
    api_prefix: str = "/api"
    openai_api_key: Optional[str] = Field(default=None)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    chroma_path: str = Field(default="data/chroma")
    otlp_endpoint: Optional[str] = Field(default=None)
    otlp_insecure: bool = Field(default=True)
    top_k: int = Field(default=3)
    api_keys: Optional[str] = Field(default=None)
    rate_limit_per_minute: int = Field(default=60)

    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
