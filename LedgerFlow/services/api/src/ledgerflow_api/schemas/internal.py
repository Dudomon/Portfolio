import uuid

from pydantic import BaseModel, Field

from ledgerflow_api.domain.fields import DocType
from ledgerflow_api.domain.lifecycle import DocumentStatus
from ledgerflow_api.schemas.documents import FieldValue


class ExtractionResult(BaseModel):
    """Posted by the worker. Unknown envelope keys are ignored (tolerant reader); unknown
    *field* keys are rejected because they would become stored data."""

    event_id: uuid.UUID
    document_id: uuid.UUID
    tenant_id: uuid.UUID
    doc_type: DocType
    model: str | None = Field(default=None, max_length=100)
    prompt_version: str | None = Field(default=None, max_length=100)
    fields: dict[str, FieldValue]
    cost_usd_micros: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class StatusReport(BaseModel):
    status: DocumentStatus
    error: str | None = Field(default=None, max_length=4000)
