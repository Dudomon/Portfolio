from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    id: Optional[str] = None
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    tenant_id: str = Field(..., example="tenant-123")
    documents: List[Document]


class ChatRequest(BaseModel):
    tenant_id: str
    question: str


class Source(BaseModel):
    id: Optional[str]
    score: float
    text: str
    metadata: Dict[str, Any]


class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
