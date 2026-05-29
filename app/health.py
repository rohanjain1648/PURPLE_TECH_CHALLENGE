"""
Health endpoint logic — what an on-call engineer checks first.
Returns STALE_FEED if a store hasn't sent an event in >10 minutes.
"""
from __future__ import annotations

import time
import structlog
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import EventORM, HealthResponse, StoreHealth

logger = structlog.get_logger(__name__)
settings = get_settings()
_start_time = time.monotonic()


async def get_health(db: AsyncSession) -> HealthResponse:
    db_status = "connected"
    stores: list[StoreHealth] = []

    try:
        # One query: most recent event timestamp per store
        stmt = (
            select(EventORM.store_id, func.max(EventORM.ingested_at).label("last_event"))
            .group_by(EventORM.store_id)
        )
        result = await db.execute(stmt)
        rows = result.all()

        now = datetime.now(tz=timezone.utc)
        stale_threshold = timedelta(minutes=settings.stale_feed_minutes)

        for row in rows:
            last_at: datetime = row.last_event
            # SQLite stores UTC but may return naive dt
            if last_at.tzinfo is None:
                last_at = last_at.replace(tzinfo=timezone.utc)
            lag = now - last_at
            status = "STALE_FEED" if lag > stale_threshold else "HEALTHY"
            stores.append(StoreHealth(store_id=row.store_id, last_event_at=last_at, status=status))

    except Exception as exc:
        logger.warning("health_db_error", error=str(exc))
        db_status = "disconnected"

    any_stale = any(s.status == "STALE_FEED" for s in stores)
    overall = "unhealthy" if db_status == "disconnected" else ("degraded" if any_stale else "healthy")

    return HealthResponse(
        status=overall,
        database=db_status,
        stores=stores,
        uptime_seconds=round(time.monotonic() - _start_time, 1),
        version="1.0.0",
        checked_at=datetime.now(tz=timezone.utc),
    )
