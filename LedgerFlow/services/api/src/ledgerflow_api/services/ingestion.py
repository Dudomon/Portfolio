"""Worker-facing use cases behind the internal endpoints.

These run in a system session (no tenant scope) because the status callback carries no tenant
id; every query therefore filters on ids explicitly.
"""

import uuid
from datetime import UTC, datetime

import structlog
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ledgerflow_api.db.models import Extraction, ProcessedEvent, Tenant
from ledgerflow_api.domain.fields import (
    FieldValueError,
    assess_fields,
    overall_confidence,
    parse_field_key,
)
from ledgerflow_api.domain.lifecycle import (
    WORKER_REPORTABLE_STATUSES,
    DocumentStatus,
    review_outcome,
)
from ledgerflow_api.errors import NotFoundError, UnprocessableError
from ledgerflow_api.schemas.internal import ExtractionResult, StatusReport
from ledgerflow_api.services.audit import Diff
from ledgerflow_api.services.documents import attach_extraction, lock_document, transition

EXTRACTION_COMPLETED = "extraction.completed"

logger = structlog.get_logger(__name__)


async def _claim_event(
    session: AsyncSession, event_id: uuid.UUID, tenant_id: uuid.UUID, event_type: str
) -> bool:
    """Record ``event_id`` as processed; ``False`` means a previous delivery already did.

    The insert happens in the same transaction as the side effects, so a crash after this
    point rolls the claim back too and the redelivery is processed normally.
    """
    claimed = await session.scalar(
        insert(ProcessedEvent)
        .values(event_id=event_id, tenant_id=tenant_id, event_type=event_type)
        .on_conflict_do_nothing(index_elements=[ProcessedEvent.event_id])
        .returning(ProcessedEvent.event_id)
    )
    return claimed is not None


async def ingest_extraction(session: AsyncSession, result: ExtractionResult) -> None:
    unknown = []
    for key in result.fields:
        try:
            parse_field_key(key)
        except FieldValueError:
            unknown.append(key)
    if unknown:
        raise UnprocessableError(
            "Extraction contains fields outside the contract.",
            extensions={"unknown_fields": unknown},
        )

    if not await _claim_event(session, result.event_id, result.tenant_id, EXTRACTION_COMPLETED):
        logger.info("extraction_duplicate_ignored", event_id=str(result.event_id))
        return

    document = await lock_document(session, result.document_id, tenant_id=result.tenant_id)
    tenant = await session.get(Tenant, result.tenant_id)
    if tenant is None:
        raise NotFoundError(f"Tenant {result.tenant_id} does not exist.")

    assessment = assess_fields(
        {key: field.value for key, field in result.fields.items()}, result.doc_type
    )
    fields = {
        key: {
            "value": assessment.values[key],
            "confidence": field.confidence,
            "source": field.source,
        }
        for key, field in result.fields.items()
    }
    confidence = overall_confidence(field.confidence for field in result.fields.values())
    outcome = review_outcome(
        overall_confidence=confidence,
        threshold=float(tenant.auto_approve_threshold),
        has_issues=bool(assessment.issues),
    )

    # A worker may skip the explicit "processing" callback; the extraction itself proves it ran.
    if document.status == DocumentStatus.QUEUED:
        transition(session, document, DocumentStatus.PROCESSING, actor_id=None)
    transition(session, document, outcome, actor_id=None)

    extraction = Extraction(
        tenant_id=document.tenant_id,
        document_id=document.id,
        version=(document.extraction_version or 0) + 1,
        model=result.model,
        prompt_version=result.prompt_version,
        fields=fields,
        overall_confidence=confidence,
        cost_usd_micros=result.cost_usd_micros,
        latency_ms=result.latency_ms,
        created_by_id=None,
    )
    session.add(extraction)
    attach_extraction(document, extraction)
    document.doc_type = result.doc_type
    document.review_required = outcome == DocumentStatus.NEEDS_REVIEW
    document.processed_at = datetime.now(UTC)
    await session.commit()
    logger.info(
        "extraction_ingested",
        document_id=str(document.id),
        outcome=outcome.value,
        overall_confidence=confidence,
        issues=[f"{issue.key}: {issue.message}" for issue in assessment.issues],
    )


async def report_status(
    session: AsyncSession, document_id: uuid.UUID, report: StatusReport
) -> None:
    if report.status not in WORKER_REPORTABLE_STATUSES:
        allowed = sorted(s.value for s in WORKER_REPORTABLE_STATUSES)
        raise UnprocessableError(f"Workers may only report {allowed}.")

    document = await lock_document(session, document_id)
    if document.status == report.status:
        # Redelivered callbacks are expected with at-least-once messaging.
        return
    failed = report.status == DocumentStatus.FAILED
    extra_diff: Diff | None = (
        {"last_error": {"before": document.last_error, "after": report.error}} if failed else None
    )
    transition(session, document, report.status, actor_id=None, extra_diff=extra_diff)
    if failed:
        document.last_error = report.error
    await session.commit()
