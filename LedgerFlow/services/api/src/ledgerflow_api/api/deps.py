import hmac
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from ledgerflow_api.config import Settings
from ledgerflow_api.db.session import SessionFactory, system_session, tenant_session
from ledgerflow_api.domain.roles import Role
from ledgerflow_api.errors import ForbiddenError, UnauthorizedError
from ledgerflow_api.security import InvalidTokenError, Principal, TokenService
from ledgerflow_api.services.documents import DocumentService
from ledgerflow_api.services.storage import ObjectStorage


@dataclass(frozen=True, slots=True)
class AppResources:
    """Process-wide dependencies created in the lifespan and shared by all requests."""

    settings: Settings
    engine: AsyncEngine
    session_factory: SessionFactory
    tokens: TokenService
    storage: ObjectStorage


def get_resources(request: Request) -> AppResources:
    resources: AppResources = request.app.state.resources
    return resources


Resources = Annotated[AppResources, Depends(get_resources)]


def get_storage(resources: Resources) -> ObjectStorage:
    return resources.storage


Storage = Annotated[ObjectStorage, Depends(get_storage)]

_user_bearer = HTTPBearer(auto_error=False, description="User access token from /auth/token")
_service_bearer = HTTPBearer(
    auto_error=False, scheme_name="ServiceToken", description="Shared internal service token"
)


async def get_principal(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_user_bearer)],
    resources: Resources,
) -> Principal:
    if credentials is None:
        raise UnauthorizedError()
    try:
        return await resources.tokens.verify(credentials.credentials)
    except InvalidTokenError:
        raise UnauthorizedError("The access token is invalid or expired.") from None


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]


def require_role(required: Role) -> Callable[[Principal], Awaitable[Principal]]:
    async def guard(principal: CurrentPrincipal) -> Principal:
        if not principal.role.satisfies(required):
            raise ForbiddenError(f"This operation requires the '{required}' role.")
        return principal

    return guard


ReviewerPrincipal = Annotated[Principal, Depends(require_role(Role.REVIEWER))]
AdminPrincipal = Annotated[Principal, Depends(require_role(Role.ADMIN))]


async def get_tenant_session(
    principal: CurrentPrincipal, resources: Resources
) -> AsyncIterator[AsyncSession]:
    async with tenant_session(resources.session_factory, principal.tenant_id) as session:
        yield session


async def get_system_session(resources: Resources) -> AsyncIterator[AsyncSession]:
    async with system_session(resources.session_factory) as session:
        yield session


TenantSession = Annotated[AsyncSession, Depends(get_tenant_session)]
SystemSession = Annotated[AsyncSession, Depends(get_system_session)]


def get_document_service(session: TenantSession, principal: CurrentPrincipal) -> DocumentService:
    return DocumentService(session, principal)


Documents = Annotated[DocumentService, Depends(get_document_service)]


async def require_service_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_service_bearer)],
    resources: Resources,
) -> None:
    expected = resources.settings.internal_service_token.get_secret_value().encode()
    presented = credentials.credentials.encode() if credentials else b""
    # Constant-time comparison so response timing doesn't leak how much of the token matched.
    if not hmac.compare_digest(presented, expected):
        raise UnauthorizedError("A valid service token is required.")
