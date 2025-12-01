from collections import defaultdict
from datetime import datetime, timedelta
from fastapi import Depends, Header, HTTPException, status

from .settings import Settings, get_settings


class ApiKeyAuth:
    def __init__(self, settings: Settings):
        self.valid_keys = {k.strip() for k in (settings.api_keys or "").split(",") if k.strip()}

    def __call__(self, x_api_key: str = Header(None)):
        if self.valid_keys and x_api_key not in self.valid_keys:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")


class RateLimiter:
    def __init__(self, limit_per_minute: int):
        self.limit = limit_per_minute
        self.bucket = defaultdict(list)

    def __call__(self, x_api_key: str = Header("anonymous")):
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=1)
        self.bucket[x_api_key] = [ts for ts in self.bucket[x_api_key] if ts > window_start]
        if len(self.bucket[x_api_key]) >= self.limit:
            raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="Rate limit exceeded")
        self.bucket[x_api_key].append(now)


def get_auth(settings: Settings = Depends(get_settings)) -> ApiKeyAuth:
    return ApiKeyAuth(settings)


def get_rate_limiter(settings: Settings = Depends(get_settings)) -> RateLimiter:
    return RateLimiter(settings.rate_limit_per_minute)
