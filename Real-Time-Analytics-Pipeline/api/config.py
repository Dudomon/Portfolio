"""Configuration settings"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = "kafka:29092"
    KAFKA_TOPIC: str = "ecommerce-events"
    
    # ClickHouse
    CLICKHOUSE_HOST: str = "clickhouse"
    CLICKHOUSE_PORT: int = 8123
    CLICKHOUSE_DB: str = "analytics"
    CLICKHOUSE_USER: str = "admin"
    CLICKHOUSE_PASSWORD: str = "admin123"
    
    # Redis
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8080
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    class Config:
        env_file = ".env"


settings = Settings()
