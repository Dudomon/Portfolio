from fastapi import APIRouter

from ledgerflow_api.api.deps import CurrentPrincipal, TenantSession
from ledgerflow_api.db.models import User
from ledgerflow_api.errors import UnauthorizedError
from ledgerflow_api.schemas.users import UserOut

router = APIRouter(tags=["users"])


@router.get("/me")
async def read_me(principal: CurrentPrincipal, session: TenantSession) -> UserOut:
    user = await session.get(User, principal.user_id)
    if user is None or not user.is_active:
        # A valid signature for a user that no longer exists is still not a valid identity.
        raise UnauthorizedError("The account behind this token is no longer active.")
    return UserOut.model_validate(user)
