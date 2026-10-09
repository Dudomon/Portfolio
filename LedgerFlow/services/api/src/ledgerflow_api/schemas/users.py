import uuid

from ledgerflow_api.domain.roles import Role
from ledgerflow_api.schemas.common import ApiModel


class UserOut(ApiModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    email: str
    name: str
    role: Role
