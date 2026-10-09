import uuid
from datetime import datetime

from pydantic import BaseModel, JsonValue

from ledgerflow_api.schemas.common import ActorRef, ApiModel


class DiffEntry(BaseModel):
    before: JsonValue
    after: JsonValue


class AuditEventOut(ApiModel):
    id: uuid.UUID
    actor: ActorRef | None
    action: str
    diff: dict[str, DiffEntry] | None
    request_id: str | None
    created_at: datetime
