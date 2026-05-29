"""
Real-time metric computation for GET /stores/{id}/metrics and /heatmap.
All queries read directly from the events and sessions tables — no stale cache.
"""
from __future__ import annotations

import structlog
from datetime import date, datetime, timezone
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    EventORM,
    EventType,
    HeatmapResponse,
    HeatmapZone,
    SessionORM,
    StoreMetrics,
    ZoneDwellMetric,
)

logger = structlog.get_logger(__name__)


async def get_store_metrics(store_id: str, db: AsyncSession, target_date: Optional[date] = None) -> StoreMetrics:
    today = target_date or date.today()
    day_start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    day_end = datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=timezone.utc)

    # Unique non-staff visitors with ENTRY today
    unique_visitors = await _count_unique_visitors(store_id, day_start, day_end, db)

    # Conversion rate
    conversion_rate = await _compute_conversion_rate(store_id, day_start, day_end, db)

    # Avg dwell per zone
    zone_dwells = await _avg_dwell_per_zone(store_id, day_start, day_end, db)

    # Current queue depth (active visitors in BILLING zone)
    queue_depth = await _current_queue_depth(store_id, db)

    # Abandonment rate
    abandonment_rate = await _compute_abandonment_rate(store_id, day_start, day_end, db)

    # Active visitors right now (ENTRY without EXIT in last 2 hours)
    active_visitors = await _count_active_visitors(store_id, db)

    return StoreMetrics(
        store_id=store_id,
        date=str(today),
        unique_visitors=unique_visitors,
        conversion_rate=round(conversion_rate, 4),
        avg_dwell_per_zone=zone_dwells,
        current_queue_depth=queue_depth,
        abandonment_rate=round(abandonment_rate, 4),
        active_visitors=active_visitors,
        computed_at=datetime.now(tz=timezone.utc),
    )


async def get_heatmap(store_id: str, db: AsyncSession, target_date: Optional[date] = None) -> HeatmapResponse:
    today = target_date or date.today()
    day_start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    day_end = datetime(today.year, today.month, today.day, 23, 59, 59, tzinfo=timezone.utc)

    # Zone visit counts (distinct sessions per zone)
    stmt_visits = (
        select(EventORM.zone_id, func.count(func.distinct(EventORM.visitor_id)).label("visit_count"))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == EventType.ZONE_ENTER.value,
            EventORM.is_staff == False,  # noqa: E712
            EventORM.timestamp.between(day_start, day_end),
            EventORM.zone_id.isnot(None),
        )
        .group_by(EventORM.zone_id)
    )
    visit_result = await db.execute(stmt_visits)
    visit_counts: dict[str, int] = {row.zone_id: row.visit_count for row in visit_result}

    # Avg dwell per zone from ZONE_DWELL events
    stmt_dwell = (
        select(EventORM.zone_id, func.avg(EventORM.dwell_ms).label("avg_dwell"))
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == EventType.ZONE_DWELL.value,
            EventORM.is_staff == False,  # noqa: E712
            EventORM.timestamp.between(day_start, day_end),
            EventORM.zone_id.isnot(None),
        )
        .group_by(EventORM.zone_id)
    )
    dwell_result = await db.execute(stmt_dwell)
    avg_dwells: dict[str, float] = {row.zone_id: float(row.avg_dwell or 0) for row in dwell_result}

    all_zones = set(visit_counts) | set(avg_dwells)
    if not all_zones:
        return HeatmapResponse(
            store_id=store_id, date=str(today), zones=[], computed_at=datetime.now(tz=timezone.utc)
        )

    # Normalise visit_count 0-100
    max_visits = max((visit_counts.get(z, 0) for z in all_zones), default=1) or 1
    min_visits = min((visit_counts.get(z, 0) for z in all_zones), default=0)

    zones: list[HeatmapZone] = []
    for zone_id in sorted(all_zones):
        count = visit_counts.get(zone_id, 0)
        dwell = avg_dwells.get(zone_id, 0.0)
        normalized = ((count - min_visits) / (max_visits - min_visits)) * 100.0 if max_visits != min_visits else 50.0
        confidence = "HIGH" if count >= 20 else "LOW"
        zones.append(HeatmapZone(
            zone_id=zone_id,
            visit_count=count,
            avg_dwell_ms=round(dwell, 1),
            normalized_score=round(normalized, 1),
            data_confidence=confidence,
        ))

    return HeatmapResponse(
        store_id=store_id,
        date=str(today),
        zones=zones,
        computed_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

async def _count_unique_visitors(store_id: str, start: datetime, end: datetime, db: AsyncSession) -> int:
    stmt = select(func.count(func.distinct(EventORM.visitor_id))).where(
        EventORM.store_id == store_id,
        EventORM.event_type == EventType.ENTRY.value,
        EventORM.is_staff == False,  # noqa: E712
        EventORM.timestamp.between(start, end),
    )
    result = await db.execute(stmt)
    return result.scalar_one() or 0


async def _compute_conversion_rate(store_id: str, start: datetime, end: datetime, db: AsyncSession) -> float:
    """Conversion = sessions with converted=True / total sessions today."""
    stmt_total = select(func.count()).where(
        SessionORM.store_id == store_id,
        SessionORM.entry_time.between(start, end),
    )
    stmt_converted = select(func.count()).where(
        SessionORM.store_id == store_id,
        SessionORM.entry_time.between(start, end),
        SessionORM.converted == True,  # noqa: E712
    )
    total = (await db.execute(stmt_total)).scalar_one() or 0
    converted = (await db.execute(stmt_converted)).scalar_one() or 0
    return converted / total if total > 0 else 0.0


async def _avg_dwell_per_zone(store_id: str, start: datetime, end: datetime, db: AsyncSession) -> list[ZoneDwellMetric]:
    stmt = (
        select(
            EventORM.zone_id,
            func.avg(EventORM.dwell_ms).label("avg_dwell"),
            func.count(func.distinct(EventORM.visitor_id)).label("visitor_count"),
        )
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == EventType.ZONE_DWELL.value,
            EventORM.is_staff == False,  # noqa: E712
            EventORM.timestamp.between(start, end),
            EventORM.zone_id.isnot(None),
        )
        .group_by(EventORM.zone_id)
    )
    result = await db.execute(stmt)
    return [
        ZoneDwellMetric(zone_id=row.zone_id, avg_dwell_ms=round(float(row.avg_dwell or 0), 1), visitor_count=row.visitor_count)
        for row in result
    ]


async def _current_queue_depth(store_id: str, db: AsyncSession) -> int:
    """Count open sessions currently in billing zone (entered but not exited billing)."""
    stmt = select(func.count()).where(
        SessionORM.store_id == store_id,
        SessionORM.billing_entry_time.isnot(None),
        SessionORM.exit_time.is_(None),
        SessionORM.converted == False,  # noqa: E712
        SessionORM.queue_abandoned == False,  # noqa: E712
    )
    result = await db.execute(stmt)
    return result.scalar_one() or 0


async def _compute_abandonment_rate(store_id: str, start: datetime, end: datetime, db: AsyncSession) -> float:
    stmt_joined = select(func.count()).where(
        SessionORM.store_id == store_id,
        SessionORM.entry_time.between(start, end),
        SessionORM.queue_joined == True,  # noqa: E712
    )
    stmt_abandoned = select(func.count()).where(
        SessionORM.store_id == store_id,
        SessionORM.entry_time.between(start, end),
        SessionORM.queue_abandoned == True,  # noqa: E712
    )
    joined = (await db.execute(stmt_joined)).scalar_one() or 0
    abandoned = (await db.execute(stmt_abandoned)).scalar_one() or 0
    return abandoned / joined if joined > 0 else 0.0


async def _count_active_visitors(store_id: str, db: AsyncSession) -> int:
    """Sessions with ENTRY but no EXIT (visitor still in store)."""
    stmt = select(func.count()).where(
        SessionORM.store_id == store_id,
        SessionORM.entry_time.isnot(None),
        SessionORM.exit_time.is_(None),
    )
    result = await db.execute(stmt)
    return result.scalar_one() or 0
