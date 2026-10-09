from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Query

from ledgerflow_api.api.deps import CurrentPrincipal, TenantSession
from ledgerflow_api.errors import UnprocessableError
from ledgerflow_api.schemas.stats import StatsOut
from ledgerflow_api.services.stats import compute_stats

router = APIRouter(tags=["stats"])

DEFAULT_WINDOW = timedelta(days=30)


def _as_utc(moment: datetime) -> datetime:
    return moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment


@router.get("/stats")
async def read_stats(
    _principal: CurrentPrincipal,
    session: TenantSession,
    start: Annotated[datetime | None, Query(alias="from")] = None,
    end: Annotated[datetime | None, Query(alias="to")] = None,
) -> StatsOut:
    end = _as_utc(end) if end else datetime.now(UTC)
    start = _as_utc(start) if start else end - DEFAULT_WINDOW
    if start >= end:
        raise UnprocessableError("'from' must be earlier than 'to'.")
    return await compute_stats(session, start=start, end=end)
