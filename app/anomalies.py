"""
Anomaly detection — runs as a background task every 30 s.
Writes to the anomalies table; the API endpoint reads from it.
"""
from __future__ import annotations

import asyncio
import structlog
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db_session
from app.models import AnomalyORM, AnomaliesResponse, AnomalyResponse, EventORM, EventType, SessionORM

logger = structlog.get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------

async def anomaly_detection_loop() -> None:
    """Runs forever, re-evaluating anomalies every 30 seconds."""
    logger.info("anomaly_detection_loop_started")
    while True:
        try:
            async with get_db_session() as db:
                store_ids = await _get_all_store_ids(db)
                for store_id in store_ids:
                    await _detect_for_store(store_id, db)
        except asyncio.CancelledError:
            logger.info("anomaly_detection_loop_cancelled")
            return
        except Exception as exc:
            logger.warning("anomaly_detection_error", error=str(exc))
        await asyncio.sleep(30)


# ---------------------------------------------------------------------------
# Public API (called by route handler)
# ---------------------------------------------------------------------------

async def get_active_anomalies(store_id: str, db: AsyncSession) -> AnomaliesResponse:
    stmt = (
        select(AnomalyORM)
        .where(AnomalyORM.store_id == store_id, AnomalyORM.resolved_at.is_(None))
        .order_by(AnomalyORM.detected_at.desc())
    )
    result = await db.execute(stmt)
    anomalies = result.scalars().all()
    return AnomaliesResponse(
        store_id=store_id,
        active_anomalies=[_to_response(a) for a in anomalies],
        computed_at=datetime.now(tz=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Detection logic (one store at a time)
# ---------------------------------------------------------------------------

async def _detect_for_store(store_id: str, db: AsyncSession) -> None:
    await _detect_queue_spike(store_id, db)
    await _detect_conversion_drop(store_id, db)
    await _detect_dead_zones(store_id, db)
    await _detect_high_entry_rate(store_id, db)
    # Auto-resolve stale anomalies
    await _resolve_cleared_anomalies(store_id, db)


async def _detect_queue_spike(store_id: str, db: AsyncSession) -> None:
    """Open sessions in billing zone right now."""
    stmt = select(func.count()).where(
        SessionORM.store_id == store_id,
        SessionORM.billing_entry_time.isnot(None),
        SessionORM.exit_time.is_(None),
        SessionORM.converted == False,  # noqa: E712
        SessionORM.queue_abandoned == False,  # noqa: E712
    )
    depth = (await db.execute(stmt)).scalar_one() or 0

    if depth >= settings.queue_critical_threshold:
        severity, action = "CRITICAL", f"Deploy additional billing staff immediately. Current queue: {depth}"
    elif depth >= settings.queue_spike_threshold:
        severity, action = "WARN", f"Queue depth is {depth}. Consider opening another billing counter."
    else:
        return  # No spike — will be resolved in _resolve_cleared_anomalies

    await _upsert_anomaly(
        store_id=store_id,
        anomaly_type="BILLING_QUEUE_SPIKE",
        severity=severity,
        suggested_action=action,
        metadata={"queue_depth": depth},
        db=db,
    )


async def _detect_conversion_drop(store_id: str, db: AsyncSession) -> None:
    """Compare today's conversion rate against the 7-day average."""
    now = datetime.now(tz=timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Today's conversion
    today_rate = await _conversion_rate_for_period(store_id, today_start, now, db)
    if today_rate is None:
        return  # Not enough data

    # 7-day historical average (excluding today)
    week_ago = today_start - timedelta(days=7)
    hist_rate = await _conversion_rate_for_period(store_id, week_ago, today_start, db)
    if hist_rate is None or hist_rate == 0:
        return

    drop_pct = (hist_rate - today_rate) / hist_rate
    if drop_pct >= settings.conversion_drop_pct:
        severity = "CRITICAL" if drop_pct >= 0.35 else "WARN"
        await _upsert_anomaly(
            store_id=store_id,
            anomaly_type="CONVERSION_DROP",
            severity=severity,
            suggested_action=(
                f"Conversion rate dropped {drop_pct:.0%} vs 7-day avg "
                f"({today_rate:.1%} vs {hist_rate:.1%}). Review staffing and in-store promotions."
            ),
            metadata={"today_rate": today_rate, "historical_avg": hist_rate, "drop_pct": drop_pct},
            db=db,
        )


async def _detect_dead_zones(store_id: str, db: AsyncSession) -> None:
    """Any zone that hasn't seen a ZONE_ENTER in the last 30 minutes."""
    threshold = datetime.now(tz=timezone.utc) - timedelta(minutes=settings.dead_zone_minutes)

    # Zones that had activity in the last 7 days (so we know they normally get traffic)
    week_ago = datetime.now(tz=timezone.utc) - timedelta(days=7)
    stmt_active = (
        select(EventORM.zone_id)
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == EventType.ZONE_ENTER.value,
            EventORM.is_staff == False,  # noqa: E712
            EventORM.timestamp > week_ago,
            EventORM.zone_id.isnot(None),
        )
        .distinct()
    )
    normally_active = {row[0] for row in (await db.execute(stmt_active))}

    # Which of those had no visit in the dead-zone window?
    stmt_recent = (
        select(EventORM.zone_id)
        .where(
            EventORM.store_id == store_id,
            EventORM.event_type == EventType.ZONE_ENTER.value,
            EventORM.is_staff == False,  # noqa: E712
            EventORM.timestamp > threshold,
            EventORM.zone_id.isnot(None),
        )
        .distinct()
    )
    recently_active = {row[0] for row in (await db.execute(stmt_recent))}

    dead_zones = normally_active - recently_active
    for zone_id in dead_zones:
        await _upsert_anomaly(
            store_id=store_id,
            anomaly_type="DEAD_ZONE",
            severity="INFO",
            suggested_action=f"Zone '{zone_id}' has had no customer visits in {settings.dead_zone_minutes} minutes. Check signage and camera feed.",
            metadata={"zone_id": zone_id, "threshold_minutes": settings.dead_zone_minutes},
            db=db,
        )


async def _detect_high_entry_rate(store_id: str, db: AsyncSession) -> None:
    """
    Detect a sudden footfall surge: if the last 10-minute entry count is more
    than 3× the average 10-minute entry rate over the past hour, raise WARN.
    Useful for alerting staff before billing queues spike.
    """
    now = datetime.now(tz=timezone.utc)
    window_start = now - timedelta(minutes=10)
    hour_ago = now - timedelta(hours=1)

    recent_stmt = select(func.count()).where(
        EventORM.store_id == store_id,
        EventORM.event_type == EventType.ENTRY.value,
        EventORM.is_staff == False,  # noqa: E712
        EventORM.timestamp > window_start,
    )
    recent_count = (await db.execute(recent_stmt)).scalar_one() or 0

    hour_stmt = select(func.count()).where(
        EventORM.store_id == store_id,
        EventORM.event_type == EventType.ENTRY.value,
        EventORM.is_staff == False,  # noqa: E712
        EventORM.timestamp.between(hour_ago, window_start),
    )
    hour_count = (await db.execute(hour_stmt)).scalar_one() or 0

    # Need at least 5 entries in the prior hour to establish a baseline
    if hour_count < 5:
        return

    baseline_per_10min = hour_count / 5.0  # 5 non-overlapping 10-min windows in an hour
    if baseline_per_10min <= 0:
        return

    ratio = recent_count / baseline_per_10min
    if ratio >= 3.0:
        severity = "CRITICAL" if ratio >= 5.0 else "WARN"
        await _upsert_anomaly(
            store_id=store_id,
            anomaly_type="HIGH_ENTRY_RATE",
            severity=severity,
            suggested_action=(
                f"Footfall surge detected: {recent_count} entries in last 10 min "
                f"({ratio:.1f}× baseline of {baseline_per_10min:.1f}/10 min). "
                "Open additional billing counters and deploy floor staff."
            ),
            metadata={"recent_count": recent_count, "baseline_per_10min": baseline_per_10min, "ratio": ratio},
            db=db,
        )


async def _resolve_cleared_anomalies(store_id: str, db: AsyncSession) -> None:
    """
    Mark BILLING_QUEUE_SPIKE as resolved when queue drops below threshold.
    DEAD_ZONE and CONVERSION_DROP are naturally re-evaluated each cycle.
    """
    now = datetime.now(tz=timezone.utc)
    depth_stmt = select(func.count()).where(
        SessionORM.store_id == store_id,
        SessionORM.billing_entry_time.isnot(None),
        SessionORM.exit_time.is_(None),
        SessionORM.converted == False,  # noqa: E712
        SessionORM.queue_abandoned == False,  # noqa: E712
    )
    depth = (await db.execute(depth_stmt)).scalar_one() or 0

    if depth < settings.queue_spike_threshold:
        await db.execute(
            update(AnomalyORM)
            .where(
                AnomalyORM.store_id == store_id,
                AnomalyORM.anomaly_type == "BILLING_QUEUE_SPIKE",
                AnomalyORM.resolved_at.is_(None),
            )
            .values(resolved_at=now)
        )

    # Resolve HIGH_ENTRY_RATE when the last 10-min window drops back to baseline
    recent_stmt = select(func.count()).where(
        EventORM.store_id == store_id,
        EventORM.event_type == EventType.ENTRY.value,
        EventORM.is_staff == False,  # noqa: E712
        EventORM.timestamp > (now - timedelta(minutes=10)),
    )
    recent_count = (await db.execute(recent_stmt)).scalar_one() or 0
    hour_stmt = select(func.count()).where(
        EventORM.store_id == store_id,
        EventORM.event_type == EventType.ENTRY.value,
        EventORM.is_staff == False,  # noqa: E712
        EventORM.timestamp.between(now - timedelta(hours=1), now - timedelta(minutes=10)),
    )
    hour_count = (await db.execute(hour_stmt)).scalar_one() or 0
    baseline = (hour_count / 5.0) if hour_count >= 5 else None
    if baseline and recent_count < baseline * 3.0:
        await db.execute(
            update(AnomalyORM)
            .where(
                AnomalyORM.store_id == store_id,
                AnomalyORM.anomaly_type == "HIGH_ENTRY_RATE",
                AnomalyORM.resolved_at.is_(None),
            )
            .values(resolved_at=now)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _conversion_rate_for_period(
    store_id: str, start: datetime, end: datetime, db: AsyncSession
) -> Optional[float]:
    total = (
        await db.execute(
            select(func.count()).where(
                SessionORM.store_id == store_id,
                SessionORM.entry_time.between(start, end),
            )
        )
    ).scalar_one() or 0
    if total < 5:
        return None
    converted = (
        await db.execute(
            select(func.count()).where(
                SessionORM.store_id == store_id,
                SessionORM.entry_time.between(start, end),
                SessionORM.converted == True,  # noqa: E712
            )
        )
    ).scalar_one() or 0
    return converted / total


async def _upsert_anomaly(
    store_id: str,
    anomaly_type: str,
    severity: str,
    suggested_action: str,
    metadata: dict,
    db: AsyncSession,
) -> None:
    # Check if an active anomaly of this type already exists
    stmt = select(AnomalyORM).where(
        AnomalyORM.store_id == store_id,
        AnomalyORM.anomaly_type == anomaly_type,
        AnomalyORM.resolved_at.is_(None),
    )
    # For DEAD_ZONE, also match on zone_id in metadata
    result = await db.execute(stmt)
    existing = result.scalars().all()

    if anomaly_type == "DEAD_ZONE":
        zone_id = metadata.get("zone_id")
        existing = [a for a in existing if (a.anomaly_metadata or {}).get("zone_id") == zone_id]

    if existing:
        # Update severity if it escalated
        for a in existing:
            a.severity = severity
            a.suggested_action = suggested_action
            a.anomaly_metadata = metadata
    else:
        db.add(AnomalyORM(
            store_id=store_id,
            anomaly_type=anomaly_type,
            severity=severity,
            suggested_action=suggested_action,
            anomaly_metadata=metadata,
        ))


async def _get_all_store_ids(db: AsyncSession) -> list[str]:
    stmt = select(EventORM.store_id).distinct()
    result = await db.execute(stmt)
    return [row[0] for row in result]


def _to_response(a: AnomalyORM) -> AnomalyResponse:
    return AnomalyResponse(
        id=a.id,
        store_id=a.store_id,
        anomaly_type=a.anomaly_type,
        severity=a.severity,
        detected_at=a.detected_at,
        suggested_action=a.suggested_action or "",
        metadata=a.anomaly_metadata,
    )
