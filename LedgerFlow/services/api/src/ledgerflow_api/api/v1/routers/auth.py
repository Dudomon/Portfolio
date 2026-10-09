from fastapi import APIRouter
from sqlalchemy import func, select
from starlette.concurrency import run_in_threadpool

from ledgerflow_api.api.deps import Resources, SystemSession
from ledgerflow_api.db.models import User
from ledgerflow_api.errors import UnauthorizedError
from ledgerflow_api.schemas.auth import TokenRequest, TokenResponse
from ledgerflow_api.security import hash_password, needs_rehash, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token")
async def issue_token(
    body: TokenRequest, session: SystemSession, resources: Resources
) -> TokenResponse:
    # Login precedes knowing the tenant, so this is one of the few cross-tenant lookups.
    user = await session.scalar(
        select(User).where(func.lower(User.email) == body.email.strip().lower())
    )
    known_hash = user.password_hash if user is not None and user.is_active else None
    # Argon2 is deliberately CPU-heavy; keep it off the event loop.
    if user is None or not await run_in_threadpool(verify_password, known_hash, body.password):
        raise UnauthorizedError("Invalid email or password.")

    if needs_rehash(user.password_hash):
        user.password_hash = await run_in_threadpool(hash_password, body.password)
        await session.commit()

    issued = resources.tokens.issue(user_id=user.id, tenant_id=user.tenant_id, role=user.role)
    return TokenResponse(access_token=issued.token, expires_in=issued.expires_in)
