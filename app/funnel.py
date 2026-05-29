"""
Conversion funnel: Entry → Zone Visit → Billing Queue → Purchase.
Session is the unit — re-entries do not double-count a visitor.
"""
from __future__ import annotations

import structlog
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    FunnelResponse,
    FunnelStage,
    SessionORM,
)

logger = structlog.get_logger(__name__)


async def get_funnel(
    store_id: str, db: AsyncSession, target_date: Optional[date] = None
) -> FunnelResponse:
    today = target_date or date.today()
    day_start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    day_end = datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=timezone.utc)

    # Count distinct visitor_ids at each stage (unique visitors, not sessions)
    # Stage 1 – Entry
    entries = await _count_distinct_visitors(store_id, day_start, day_end, db, stage="entry")
    # Stage 2 – Zone Visit (visited at least one product zone)
    zone_visits = await _count_distinct_visitors(store_id, day_start, day_end, db, stage="zone")
    # Stage 3 – Billing Queue (joined billing queue)
    billing = await _count_distinct_visitors(store_id, day_start, day_end, db, stage="billing")
    # Stage 4 – Purchase (converted)
    purchases = await _count_distinct_visitors(store_id, day_start, day_end, db, stage="purchase")

    def drop_off(prev: int, curr: int) -> float:
        if prev == 0:
            return 0.0
        return round((prev - curr) / prev * 100, 1)

    stages = [
        FunnelStage(stage="Entry", count=entries, drop_off_pct=0.0),
        FunnelStage(stage="Zone Visit", count=zone_visits, drop_off_pct=drop_off(entries, zone_visits)),
        FunnelStage(stage="Billing Queue", count=billing, drop_off_pct=drop_off(zone_visits, billing)),
        FunnelStage(stage="Purchase", count=purchases, drop_off_pct=drop_off(billing, purchases)),
    ]

    return FunnelResponse(
        store_id=store_id,
        date=str(today),
        stages=stages,
        computed_at=datetime.now(tz=timezone.utc),
    )


async def _count_distinct_visitors(
    store_id: str,
    start: datetime,
    end: datetime,
    db: AsyncSession,
    stage: str,
) -> int:
    """Count distinct visitor_ids at each funnel stage (de-duplicates re-entries)."""
    base = select(func.count(func.distinct(SessionORM.visitor_id))).where(
        SessionORM.store_id == store_id,
        SessionORM.entry_time.between(start, end),
    )

    if stage == "entry":
        stmt = base
    elif stage == "zone":
        # zones_visited is a JSON list; visitor visited at least 1 zone
        # SQLite: json_array_length, PostgreSQL: jsonb_array_length
        # Use Python-side filtering to stay DB-agnostic
        stmt_all = select(SessionORM).where(
            SessionORM.store_id == store_id,
            SessionORM.entry_time.between(start, end),
        )
        result = await db.execute(stmt_all)
        sessions = result.scalars().all()
        visitor_ids = {
            s.visitor_id for s in sessions if s.zones_visited and len(s.zones_visited) > 0
        }
        return len(visitor_ids)
    elif stage == "billing":
        stmt_all = select(SessionORM).where(
            SessionORM.store_id == store_id,
            SessionORM.entry_time.between(start, end),
            SessionORM.queue_joined == True,  # noqa: E712
        )
        result = await db.execute(stmt_all)
        return len({s.visitor_id for s in result.scalars()})
    elif stage == "purchase":
        stmt_all = select(SessionORM).where(
            SessionORM.store_id == store_id,
            SessionORM.entry_time.between(start, end),
            SessionORM.converted == True,  # noqa: E712
        )
        result = await db.execute(stmt_all)
        return len({s.visitor_id for s in result.scalars()})
    else:
        stmt = base

    result = await db.execute(stmt)
    return result.scalar_one() or 0
