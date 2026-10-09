"""Password hashing and JWT issuing/verification."""

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from starlette.concurrency import run_in_threadpool

from ledgerflow_api.config import Settings
from ledgerflow_api.domain.roles import Role

_hasher = PasswordHasher()
# Verifying against a real hash when the email is unknown keeps login timing uniform, so
# response latency doesn't reveal which addresses have accounts.
_DUMMY_HASH = _hasher.hash("ledgerflow-timing-equalizer")

_REQUIRED_CLAIMS = ["exp", "iat", "sub", "tid", "role"]


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    """``None`` means the account does not exist; the same hashing work is done regardless."""
    try:
        _hasher.verify(password_hash or _DUMMY_HASH, password)
    except (VerificationError, InvalidHashError):
        return False
    return password_hash is not None


def needs_rehash(password_hash: str) -> bool:
    return _hasher.check_needs_rehash(password_hash)


@dataclass(frozen=True, slots=True)
class Principal:
    user_id: uuid.UUID
    tenant_id: uuid.UUID
    role: Role


@dataclass(frozen=True, slots=True)
class IssuedToken:
    token: str
    expires_in: int


class InvalidTokenError(Exception):
    pass


def _required[T](value: T | None, name: str) -> T:
    if value is None:
        raise RuntimeError(f"{name} is not configured")
    return value


class TokenService:
    def __init__(self, settings: Settings) -> None:
        self._algorithm = settings.jwt_algorithm
        self._ttl = timedelta(seconds=settings.access_token_ttl_seconds)
        self._jwks: jwt.PyJWKClient | None = None
        if settings.jwt_algorithm == "HS256":
            secret = _required(settings.jwt_secret, "JWT_SECRET").get_secret_value()
            self._signing_key: str = secret
            self._verification_key: str | None = secret
        else:
            private_key = _required(settings.jwt_private_key, "JWT_PRIVATE_KEY")
            self._signing_key = private_key.get_secret_value()
            self._verification_key = settings.jwt_public_key
            if settings.jwt_jwks_url:
                self._jwks = jwt.PyJWKClient(settings.jwt_jwks_url, cache_keys=True)

    def issue(self, *, user_id: uuid.UUID, tenant_id: uuid.UUID, role: Role) -> IssuedToken:
        now = datetime.now(UTC)
        claims = {
            "sub": str(user_id),
            "tid": str(tenant_id),
            "role": role.value,
            "iat": now,
            "exp": now + self._ttl,
        }
        token = jwt.encode(claims, self._signing_key, algorithm=self._algorithm)
        return IssuedToken(token=token, expires_in=int(self._ttl.total_seconds()))

    async def verify(self, token: str) -> Principal:
        try:
            key = await self._resolve_key(token)
            claims: dict[str, Any] = jwt.decode(
                token,
                key,
                algorithms=[self._algorithm],
                options={"require": _REQUIRED_CLAIMS},
            )
            return Principal(
                user_id=uuid.UUID(claims["sub"]),
                tenant_id=uuid.UUID(claims["tid"]),
                role=Role(claims["role"]),
            )
        except (jwt.PyJWTError, ValueError) as exc:
            raise InvalidTokenError(str(exc)) from exc

    async def _resolve_key(self, token: str) -> Any:
        if self._jwks is not None:
            # PyJWKClient does blocking HTTP on a cache miss.
            signing_key = await run_in_threadpool(self._jwks.get_signing_key_from_jwt, token)
            return signing_key.key
        return self._verification_key
