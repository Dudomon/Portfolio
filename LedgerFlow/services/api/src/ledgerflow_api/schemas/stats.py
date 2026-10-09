from pydantic import BaseModel

from ledgerflow_api.domain.lifecycle import DocumentStatus


class StatsOut(BaseModel):
    by_status: dict[DocumentStatus, int]
    auto_approval_rate: float
    avg_confidence: float
    p95_processing_ms: int
    llm_cost_usd: str
