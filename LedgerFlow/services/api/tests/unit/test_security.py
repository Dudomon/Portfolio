import uuid
from datetime import UTC, datetime, timedelta

import jwt
import pytest

from ledgerflow_api.domain.roles import Role
from ledgerflow_api.security import (
    InvalidTokenError,
    TokenService,
    hash_password,
    verify_password,
)
from tests.conftest import TEST_JWT_SECRET, make_settings


@pytest.fixture
def tokens() -> TokenService:
    return TokenService(make_settings())


async def test_issued_token_round_trips(tokens: TokenService) -> None:
    user_id, tenant_id = uuid.uuid4(), uuid.uuid4()
    issued = tokens.issue(user_id=user_id, tenant_id=tenant_id, role=Role.REVIEWER)
    principal = await tokens.verify(issued.token)
    assert (principal.user_id, principal.tenant_id, principal.role) == (
        user_id,
        tenant_id,
        Role.REVIEWER,
    )
    assert issued.expires_in == 900


@pytest.mark.parametrize(
    "overrides",
    [
        {"exp": datetime.now(UTC) - timedelta(seconds=1)},
        {"role": "superuser"},
        {"tid": None},
    ],
    ids=["expired", "unknown-role", "missing-tenant"],
)
async def test_rejects_bad_claims(tokens: TokenService, overrides: dict[str, object]) -> None:
    now = datetime.now(UTC)
    claims = {
        "sub": str(uuid.uuid4()),
        "tid": str(uuid.uuid4()),
        "role": "admin",
        "iat": now,
        "exp": now + timedelta(minutes=5),
    } | overrides
    token = jwt.encode({k: v for k, v in claims.items() if v is not None}, TEST_JWT_SECRET)
    with pytest.raises(InvalidTokenError):
        await tokens.verify(token)


async def test_rejects_foreign_signature(tokens: TokenService) -> None:
    foreign = TokenService(make_settings(jwt_secret="another-secret-that-is-long-enough!!"))
    issued = foreign.issue(user_id=uuid.uuid4(), tenant_id=uuid.uuid4(), role=Role.ADMIN)
    with pytest.raises(InvalidTokenError):
        await tokens.verify(issued.token)


def test_password_hashing() -> None:
    hashed = hash_password("correct horse")
    assert verify_password(hashed, "correct horse")
    assert not verify_password(hashed, "wrong horse")
    assert not verify_password(None, "correct horse")
    assert not verify_password("not-a-hash", "correct horse")


@pytest.mark.parametrize(
    ("role", "required", "allowed"),
    [
        (Role.ADMIN, Role.REVIEWER, True),
        (Role.REVIEWER, Role.REVIEWER, True),
        (Role.VIEWER, Role.REVIEWER, False),
        (Role.REVIEWER, Role.ADMIN, False),
    ],
)
def test_role_hierarchy(role: Role, required: Role, allowed: bool) -> None:
    assert role.satisfies(required) is allowed
