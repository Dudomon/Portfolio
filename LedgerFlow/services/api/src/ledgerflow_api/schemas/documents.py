import uuid
from datetime import datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, JsonValue, StringConstraints

from ledgerflow_api.db.models import Document, Extraction
from ledgerflow_api.domain.fields import FieldSource
from ledgerflow_api.domain.lifecycle import DocumentStatus
from ledgerflow_api.schemas.common import ActorRef, ApiModel, DecimalStr


class FieldValue(BaseModel):
    value: JsonValue
    confidence: float = Field(ge=0, le=1)
    source: FieldSource


class ExtractionOut(ApiModel):
    id: uuid.UUID
    version: int
    model: str | None
    prompt_version: str | None
    fields: dict[str, FieldValue]
    overall_confidence: float
    cost_usd_micros: int
    latency_ms: int
    created_at: datetime


class DocumentSummary(ApiModel):
    id: uuid.UUID
    filename: str
    doc_type: Literal["invoice", "nfe"] | None
    status: DocumentStatus
    supplier_name: str | None
    total_amount: DecimalStr | None
    currency: str | None
    overall_confidence: float | None
    uploaded_by: ActorRef
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_document(cls, document: Document) -> Self:
        return cls.model_validate(document)


class DocumentDetail(DocumentSummary):
    content_type: str
    sha256: str
    size_bytes: int
    extraction: ExtractionOut | None
    rejection_reason: str | None
    version: int

    @classmethod
    def from_parts(cls, document: Document, extraction: Extraction | None) -> Self:
        return cls(
            **DocumentSummary.from_document(document).model_dump(),
            content_type=document.content_type,
            sha256=document.sha256,
            size_bytes=document.size_bytes,
            extraction=ExtractionOut.model_validate(extraction) if extraction else None,
            rejection_reason=document.rejection_reason,
            version=document.version,
        )


class FieldCorrectionRequest(BaseModel):
    fields: dict[str, JsonValue] = Field(min_length=1)


class RejectRequest(BaseModel):
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
