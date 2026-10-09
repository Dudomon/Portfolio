"""Callbacks from the extraction worker. Not for browsers: authenticated by a shared token."""

import uuid

from fastapi import APIRouter, Depends, status

from ledgerflow_api.api.deps import SystemSession, require_service_token
from ledgerflow_api.schemas.internal import ExtractionResult, StatusReport
from ledgerflow_api.services.ingestion import ingest_extraction, report_status

router = APIRouter(
    prefix="/internal", tags=["internal"], dependencies=[Depends(require_service_token)]
)


@router.post("/extractions", status_code=status.HTTP_204_NO_CONTENT)
async def post_extraction(body: ExtractionResult, session: SystemSession) -> None:
    await ingest_extraction(session, body)


@router.post("/documents/{document_id}/status", status_code=status.HTTP_204_NO_CONTENT)
async def post_document_status(
    document_id: uuid.UUID, body: StatusReport, session: SystemSession
) -> None:
    await report_status(session, document_id, body)
