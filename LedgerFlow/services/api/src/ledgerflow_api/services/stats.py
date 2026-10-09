from datetime import datetime

from sqlalchemy import Float, case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ledgerflow_api.db.models import Document, Extraction
from ledgerflow_api.domain.lifecycle import DocumentStatus
from ledgerflow_api.domain.money import micros_to_usd
from ledgerflow_api.schemas.stats import StatsOut


async def compute_stats(session: AsyncSession, *, start: datetime, end: datetime) -> StatsOut:
    """Aggregates over documents created in ``[start, end)``; RLS limits them to the tenant.

    The aggregates rely on SQL ignoring NULLs: documents that haven't been extracted yet have no
    confidence, review decision or processing time and simply don't contribute.
    """
    in_range = (Document.created_at >= start) & (Document.created_at < end)

    status_rows = await session.execute(
        select(Document.status, func.count()).where(in_range).group_by(Document.status)
    )
    by_status = dict.fromkeys(DocumentStatus, 0)
    for status, count in status_rows:
        by_status[status] = count

    processing_ms = func.extract("epoch", Document.processed_at - Document.created_at) * 1000
    auto_approved = case(
        (Document.review_required.is_(False), 1.0),
        (Document.review_required.is_(True), 0.0),
    )
    aggregates = (
        await session.execute(
            select(
                func.avg(auto_approved),
                func.avg(Document.overall_confidence),
                func.percentile_cont(0.95).within_group(processing_ms).cast(Float),
            ).where(in_range)
        )
    ).one()
    auto_approval_rate, avg_confidence, p95_processing_ms = aggregates

    cost_micros = await session.scalar(
        select(func.coalesce(func.sum(Extraction.cost_usd_micros), 0)).where(
            (Extraction.created_at >= start) & (Extraction.created_at < end)
        )
    )

    return StatsOut(
        by_status=by_status,
        auto_approval_rate=float(auto_approval_rate or 0.0),
        avg_confidence=float(avg_confidence or 0.0),
        p95_processing_ms=round(p95_processing_ms or 0),
        llm_cost_usd=micros_to_usd(int(cost_micros or 0)),
    )
