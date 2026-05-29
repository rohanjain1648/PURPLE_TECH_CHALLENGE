"""
Event ingestion: validation, deduplication, persistence, and session state updates.
POST /events/ingest delegates entirely to ingest_events().
"""
from __future__ import annotations

import structlog
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select


def _ensure_utc(dt: datetime) -> datetime:
    """Normalise a datetime to UTC — handles SQLite returning tz-naive values."""
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models import (
    EventIn,
    EventORM,
    EventType,
    IngestResult,
    PosTransactionIn,
    PosTransactionORM,
    SessionORM,
)

logger = structlog.get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def ingest_events(
    events: list[EventIn], db: AsyncSession
) -> IngestResult:
    """
    Validate, deduplicate, persist events, and update session state.
    Partial success: malformed events are rejected but valid ones are stored.
    """
    accepted = 0
    duplicates = 0
    rejected = 0
    errors: list[dict[str, Any]] = []

    # Bulk-fetch existing event_ids to detect duplicates in one query
    event_ids = [e.event_id for e in events]
    existing_ids = await _fetch_existing_ids(event_ids, db)

    rows_to_insert: list[EventORM] = []

    for evt in events:
        if evt.event_id in existing_ids:
            duplicates += 1
            continue
        try:
            orm = _to_orm(evt)
            rows_to_insert.append(orm)
            existing_ids.add(evt.event_id)
            accepted += 1
        except Exception as exc:
            rejected += 1
            errors.append({"event_id": evt.event_id, "error": str(exc)})

    if rows_to_insert:
        db.add_all(rows_to_insert)
        await db.flush()
        # Update sessions after events are visible in the transaction
        await _update_sessions(rows_to_insert, db)

    logger.info(
        "ingest_complete",
        accepted=accepted,
        duplicates=duplicates,
        rejected=rejected,
    )
    return IngestResult(
        accepted=accepted,
        duplicates=duplicates,
        rejected=rejected,
        errors=errors,
    )


async def ingest_pos_transactions(
    transactions: list[PosTransactionIn], db: AsyncSession
) -> dict[str, int]:
    """Load POS transactions and trigger conversion correlation."""
    existing_stmt = select(PosTransactionORM.transaction_id).where(
        PosTransactionORM.transaction_id.in_([t.transaction_id for t in transactions])
    )
    existing_result = await db.execute(existing_stmt)
    existing_ids = {row[0] for row in existing_result}

    new_txns: list[PosTransactionORM] = []
    for txn in transactions:
        if txn.transaction_id not in existing_ids:
            new_txns.append(
                PosTransactionORM(
                    transaction_id=txn.transaction_id,
                    store_id=txn.store_id,
                    timestamp=txn.timestamp,
                    basket_value_inr=txn.basket_value_inr,
                )
            )

    if new_txns:
        db.add_all(new_txns)
        await db.flush()
        await _correlate_conversions(new_txns, db)

    return {"loaded": len(new_txns), "duplicates": len(transactions) - len(new_txns)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _fetch_existing_ids(event_ids: list[str], db: AsyncSession) -> set[str]:
    if not event_ids:
        return set()
    stmt = select(EventORM.event_id).where(EventORM.event_id.in_(event_ids))
    result = await db.execute(stmt)
    return {row[0] for row in result}


def _to_orm(evt: EventIn) -> EventORM:
    meta = evt.metadata.model_dump() if evt.metadata else None
    return EventORM(
        event_id=evt.event_id,
        store_id=evt.store_id,
        camera_id=evt.camera_id,
        visitor_id=evt.visitor_id,
        event_type=evt.event_type.value,
        timestamp=evt.timestamp,
        zone_id=evt.zone_id,
        dwell_ms=evt.dwell_ms,
        is_staff=evt.is_staff,
        confidence=evt.confidence,
        event_metadata=meta,
    )


async def _update_sessions(events: list[EventORM], db: AsyncSession) -> None:
    """Update session rows based on newly inserted events."""
    # Group events by (visitor_id, store_id) for efficiency
    visitor_events: dict[tuple[str, str], list[EventORM]] = {}
    for evt in events:
        if evt.is_staff:
            continue
        key = (evt.visitor_id, evt.store_id)
        visitor_events.setdefault(key, []).append(evt)

    for (visitor_id, store_id), evts in visitor_events.items():
        evts.sort(key=lambda e: e.timestamp)
        for evt in evts:
            await _apply_event_to_session(evt, db)


async def _apply_event_to_session(evt: EventORM, db: AsyncSession) -> None:
    """Update or create a session record based on a single event."""
    # Find the most recent open session for this visitor
    stmt = (
        select(SessionORM)
        .where(
            SessionORM.visitor_id == evt.visitor_id,
            SessionORM.store_id == evt.store_id,
            SessionORM.exit_time.is_(None),
        )
        .order_by(SessionORM.entry_time.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    session: SessionORM | None = result.scalar_one_or_none()

    etype = evt.event_type

    if etype == EventType.ENTRY.value:
        # Check if this is a re-entry (session exists that was exited recently)
        is_reentry = False
        if session is None:
            # Check for a recently closed session
            stmt2 = (
                select(SessionORM)
                .where(
                    SessionORM.visitor_id == evt.visitor_id,
                    SessionORM.store_id == evt.store_id,
                    SessionORM.exit_time.isnot(None),
                )
                .order_by(SessionORM.exit_time.desc())
                .limit(1)
            )
            r2 = await db.execute(stmt2)
            prev = r2.scalar_one_or_none()
            if prev and prev.exit_time:
                gap_s = (_ensure_utc(evt.timestamp) - _ensure_utc(prev.exit_time)).total_seconds()
                if 0 < gap_s <= settings.reentry_max_gap_s:
                    is_reentry = True

        if session is None:
            session_id = f"{evt.visitor_id}_{evt.store_id}_{uuid.uuid4().hex[:8]}"
            session = SessionORM(
                session_id=session_id,
                visitor_id=evt.visitor_id,
                store_id=evt.store_id,
                entry_time=evt.timestamp,
                zones_visited=[],
                is_reentry=is_reentry,
            )
            db.add(session)

    elif etype == EventType.EXIT.value:
        if session:
            session.exit_time = evt.timestamp

    elif etype in (EventType.ZONE_ENTER.value, EventType.ZONE_EXIT.value, EventType.ZONE_DWELL.value):
        if session and evt.zone_id:
            zones = list(session.zones_visited or [])
            if evt.zone_id not in zones:
                zones.append(evt.zone_id)
                session.zones_visited = zones
            if etype == EventType.ZONE_DWELL.value:
                session.total_dwell_ms = (session.total_dwell_ms or 0) + evt.dwell_ms

    elif etype == EventType.BILLING_QUEUE_JOIN.value:
        if session:
            session.billing_entry_time = evt.timestamp
            session.queue_joined = True

    elif etype == EventType.BILLING_QUEUE_ABANDON.value:
        if session:
            session.queue_abandoned = True
            session.billing_entry_time = None

    elif etype == EventType.REENTRY.value:
        # Pipeline explicitly detected re-entry; mark existing session
        if session:
            session.is_reentry = True


async def _correlate_conversions(
    new_txns: list[PosTransactionORM], db: AsyncSession
) -> None:
    """
    Mark sessions as converted if the visitor was in the billing zone
    in the 5-minute window before the POS transaction.
    """
    window_s = settings.pos_correlation_window_s
    for txn in new_txns:
        # Find sessions where billing_entry_time is within window
        stmt = select(SessionORM).where(
            SessionORM.store_id == txn.store_id,
            SessionORM.billing_entry_time.isnot(None),
            SessionORM.converted == False,  # noqa: E712
        )
        result = await db.execute(stmt)
        sessions = result.scalars().all()
        for sess in sessions:
            if sess.billing_entry_time:
                gap = (_ensure_utc(txn.timestamp) - _ensure_utc(sess.billing_entry_time)).total_seconds()
                if 0 <= gap <= window_s:
                    sess.converted = True
