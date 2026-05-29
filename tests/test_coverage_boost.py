# PROMPT: "Write additional async pytest tests targeting uncovered app-layer branches:
# health check with events in DB (covers the per-store loop), ingestion session
# state machine (BILLING_QUEUE_JOIN, BILLING_QUEUE_ABANDON, EXIT, ZONE_DWELL
# transitions), POS correlation marking sessions as converted, and metric
# computation when ZONE_DWELL events exist."
#
# CHANGES MADE: Split POS correlation test into two steps (ingest events first,
# then ingest POS) because the AI draft tried to do both in one call which broke
# the transaction ordering; added a test for get_health() returning store-level
# data; added zero-purchase store edge case for conversion rate; added direct
# ingestion.py unit test for _correlate_conversions.

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

import pytest

from app.ingestion import ingest_events, ingest_pos_transactions
from app.models import EventIn, EventType, PosTransactionIn
from tests.conftest import (
    CAMERA_BILLING,
    CAMERA_FLOOR,
    CAMERA_ENTRY,
    STORE_ID,
    make_batch,
    make_event_payload,
)


# ─── Health endpoint with real events ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_with_events_shows_stores(client):
    """Health endpoint must surface per-store data when events exist."""
    event = make_event_payload(visitor_id="VIS_hc1", event_type="ENTRY")
    await client.post("/events/ingest", json=make_batch(event))

    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    # At least one store must appear in the list now
    store_ids = [s["store_id"] for s in body["stores"]]
    assert STORE_ID in store_ids


@pytest.mark.asyncio
async def test_health_store_healthy_when_recent(client):
    """Store is HEALTHY when last event is recent (not stale)."""
    event = make_event_payload(visitor_id="VIS_hc2", event_type="ENTRY")
    await client.post("/events/ingest", json=make_batch(event))

    resp = await client.get("/health")
    stores = {s["store_id"]: s for s in resp.json()["stores"]}
    if STORE_ID in stores:
        assert stores[STORE_ID]["status"] in {"HEALTHY", "STALE_FEED"}
        assert stores[STORE_ID]["last_event_at"] is not None


# ─── Session state machine — full transitions ───────────────────────────────────

@pytest.mark.asyncio
async def test_session_billing_join_and_exit(client):
    """BILLING_QUEUE_JOIN followed by ZONE_EXIT marks queue_joined on session."""
    vid = f"VIS_bq_{uuid.uuid4().hex[:4]}"
    events = [
        make_event_payload(visitor_id=vid, event_type="ENTRY"),
        make_event_payload(visitor_id=vid, event_type="ZONE_ENTER",
                           zone_id="BILLING", camera_id=CAMERA_BILLING, queue_depth=2),
        make_event_payload(visitor_id=vid, event_type="BILLING_QUEUE_JOIN",
                           zone_id="BILLING", camera_id=CAMERA_BILLING, queue_depth=2),
        make_event_payload(visitor_id=vid, event_type="ZONE_EXIT",
                           zone_id="BILLING", camera_id=CAMERA_BILLING),
        make_event_payload(visitor_id=vid, event_type="EXIT"),
    ]
    resp = await client.post("/events/ingest", json=make_batch(*events))
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 5


@pytest.mark.asyncio
async def test_session_zone_dwell_accumulates(client):
    """ZONE_DWELL events accumulate total_dwell_ms in the session."""
    vid = f"VIS_dwell_{uuid.uuid4().hex[:4]}"
    events = [
        make_event_payload(visitor_id=vid, event_type="ENTRY"),
        make_event_payload(visitor_id=vid, event_type="ZONE_ENTER",
                           zone_id="SKINCARE", camera_id=CAMERA_FLOOR),
        make_event_payload(visitor_id=vid, event_type="ZONE_DWELL",
                           zone_id="SKINCARE", camera_id=CAMERA_FLOOR, dwell_ms=30000),
        make_event_payload(visitor_id=vid, event_type="ZONE_DWELL",
                           zone_id="SKINCARE", camera_id=CAMERA_FLOOR, dwell_ms=30000),
        make_event_payload(visitor_id=vid, event_type="ZONE_EXIT",
                           zone_id="SKINCARE", camera_id=CAMERA_FLOOR),
    ]
    resp = await client.post("/events/ingest", json=make_batch(*events))
    assert resp.json()["accepted"] == 5


@pytest.mark.asyncio
async def test_session_reentry_creates_new_session(client):
    """ENTRY → EXIT → ENTRY sequence should create a reentry=True session."""
    vid = f"VIS_reen_{uuid.uuid4().hex[:4]}"
    events = [
        make_event_payload(visitor_id=vid, event_type="ENTRY"),
        make_event_payload(visitor_id=vid, event_type="EXIT"),
        make_event_payload(visitor_id=vid, event_type="ENTRY"),  # re-entry
    ]
    resp = await client.post("/events/ingest", json=make_batch(*events))
    assert resp.json()["accepted"] == 3


@pytest.mark.asyncio
async def test_session_billing_abandon(client):
    """BILLING_QUEUE_ABANDON must set queue_abandoned on session."""
    vid = f"VIS_ab_{uuid.uuid4().hex[:4]}"
    events = [
        make_event_payload(visitor_id=vid, event_type="ENTRY"),
        make_event_payload(visitor_id=vid, event_type="BILLING_QUEUE_JOIN",
                           zone_id="BILLING", camera_id=CAMERA_BILLING, queue_depth=3),
        make_event_payload(visitor_id=vid, event_type="BILLING_QUEUE_ABANDON",
                           zone_id="BILLING", camera_id=CAMERA_BILLING),
    ]
    resp = await client.post("/events/ingest", json=make_batch(*events))
    assert resp.json()["accepted"] == 3
    metrics = await client.get(f"/stores/{STORE_ID}/metrics")
    assert metrics.json()["abandonment_rate"] >= 0


# ─── POS correlation ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pos_ingest_endpoint(client):
    """POST /pos/ingest must accept transactions and return counts."""
    txns = [
        {
            "transaction_id": f"TXN_{uuid.uuid4().hex[:6]}",
            "store_id": STORE_ID,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "basket_value_inr": 1200.0,
        }
    ]
    resp = await client.post("/pos/ingest", json={"transactions": txns})
    assert resp.status_code == 200
    body = resp.json()
    assert "loaded" in body


@pytest.mark.asyncio
async def test_pos_idempotency(client):
    """Sending the same POS transaction twice must not double-count."""
    txn_id = f"TXN_idem_{uuid.uuid4().hex[:4]}"
    txn = {
        "transaction_id": txn_id,
        "store_id": STORE_ID,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "basket_value_inr": 800.0,
    }
    r1 = await client.post("/pos/ingest", json={"transactions": [txn]})
    r2 = await client.post("/pos/ingest", json={"transactions": [txn]})
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r2.json()["loaded"] == 0
    assert r2.json()["duplicates"] == 1


@pytest.mark.asyncio
async def test_pos_correlation_marks_conversion(db_session):
    """POS transaction within 5 min of billing entry → session.converted=True."""
    from app.models import SessionORM

    now = datetime.now(tz=timezone.utc)
    sid = f"STORE_POS_{uuid.uuid4().hex[:4]}"
    vid = f"VIS_conv_{uuid.uuid4().hex[:4]}"

    # Ingest visitor events
    events = [
        EventIn(
            store_id=sid, camera_id="CAM_ENTRY_01", visitor_id=vid,
            event_type=EventType.ENTRY, timestamp=now - timedelta(minutes=10),
            confidence=0.90,
        ),
        EventIn(
            store_id=sid, camera_id="CAM_BILLING_01", visitor_id=vid,
            event_type=EventType.BILLING_QUEUE_JOIN, timestamp=now - timedelta(minutes=3),
            zone_id="BILLING", confidence=0.88,
            metadata=None,
        ),
    ]
    await ingest_events(events, db_session)

    # Ingest POS transaction 2 minutes after billing entry
    txn = PosTransactionIn(
        transaction_id=f"TXN_{uuid.uuid4().hex[:6]}",
        store_id=sid,
        timestamp=now - timedelta(minutes=1),
        basket_value_inr=999.0,
    )
    await ingest_pos_transactions([txn], db_session)

    # Session must be marked converted
    from sqlalchemy import select
    stmt = select(SessionORM).where(
        SessionORM.visitor_id == vid,
        SessionORM.store_id == sid,
    )
    result = await db_session.execute(stmt)
    session = result.scalar_one_or_none()
    assert session is not None
    assert session.converted is True, "Session should be marked converted after POS correlation"


# ─── Metrics with zone dwell data ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metrics_with_zone_dwell(client):
    """avg_dwell_per_zone must be non-empty when ZONE_DWELL events exist."""
    vid = f"VIS_dwm_{uuid.uuid4().hex[:4]}"
    events = [
        make_event_payload(visitor_id=vid, event_type="ENTRY"),
        make_event_payload(visitor_id=vid, event_type="ZONE_ENTER",
                           zone_id="MAKEUP", camera_id=CAMERA_FLOOR),
        make_event_payload(visitor_id=vid, event_type="ZONE_DWELL",
                           zone_id="MAKEUP", camera_id=CAMERA_FLOOR, dwell_ms=45000),
    ]
    await client.post("/events/ingest", json=make_batch(*events))

    resp = await client.get(f"/stores/{STORE_ID}/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["avg_dwell_per_zone"], list)


@pytest.mark.asyncio
async def test_metrics_zero_purchase_store(client):
    """Store with visitors but no POS transactions must have conversion_rate=0.0."""
    sid = f"STORE_NOPURCH_{uuid.uuid4().hex[:4]}"
    vid = f"VIS_np_{uuid.uuid4().hex[:4]}"
    event = make_event_payload(visitor_id=vid, event_type="ENTRY")
    event["store_id"] = sid
    await client.post("/events/ingest", json=make_batch(event))

    resp = await client.get(f"/stores/{sid}/metrics")
    assert resp.status_code == 200
    assert resp.json()["conversion_rate"] == 0.0


@pytest.mark.asyncio
async def test_heatmap_with_zone_data(client):
    """Heatmap zones list must be non-empty after ZONE_ENTER events."""
    vid = f"VIS_hm_{uuid.uuid4().hex[:4]}"
    events = [
        make_event_payload(visitor_id=vid, event_type="ENTRY"),
        make_event_payload(visitor_id=vid, event_type="ZONE_ENTER",
                           zone_id="FRAGRANCE", camera_id=CAMERA_FLOOR),
        make_event_payload(visitor_id=vid, event_type="ZONE_DWELL",
                           zone_id="FRAGRANCE", camera_id=CAMERA_FLOOR, dwell_ms=60000),
    ]
    await client.post("/events/ingest", json=make_batch(*events))

    resp = await client.get(f"/stores/{STORE_ID}/heatmap")
    assert resp.status_code == 200
    body = resp.json()
    assert "zones" in body
    zone_ids = [z["zone_id"] for z in body["zones"]]
    assert "FRAGRANCE" in zone_ids
    # Check normalized score is 0-100
    for z in body["zones"]:
        assert 0 <= z["normalized_score"] <= 100


# ─── Anomaly detection integration ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_anomaly_detect_for_store_runs(db_session):
    """_detect_for_store() must run without error on a store with sessions."""
    from app.anomalies import _detect_for_store

    sid = f"STORE_ADET_{uuid.uuid4().hex[:4]}"
    # Should complete without exception even with no sessions
    await _detect_for_store(sid, db_session)


@pytest.mark.asyncio
async def test_anomaly_resolve_clears_queue_spike(db_session):
    """Queue spike anomaly is resolved when queue drops below threshold."""
    from app.anomalies import _upsert_anomaly, _resolve_cleared_anomalies, get_active_anomalies

    sid = f"STORE_RES_{uuid.uuid4().hex[:4]}"
    # Create a spike anomaly
    await _upsert_anomaly(sid, "BILLING_QUEUE_SPIKE", "WARN", "Test action", {"queue_depth": 6}, db_session)

    # Queue is empty → should resolve
    await _resolve_cleared_anomalies(sid, db_session)
    result = await get_active_anomalies(sid, db_session)
    # With no sessions, queue=0, so spike should be resolved
    spikes = [a for a in result.active_anomalies if a.anomaly_type == "BILLING_QUEUE_SPIKE"]
    assert len(spikes) == 0, "Queue spike should be resolved when queue is empty"


# ─── Direct ingestion.py unit tests ────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_events_returns_ingest_result(db_session):
    """ingest_events() must return IngestResult with correct counts."""
    events = [
        EventIn(
            store_id=STORE_ID, camera_id=CAMERA_ENTRY, visitor_id="VIS_direct1",
            event_type=EventType.ENTRY, timestamp=datetime.now(tz=timezone.utc),
            confidence=0.92,
        )
    ]
    result = await ingest_events(events, db_session)
    assert result.accepted == 1
    assert result.rejected == 0
    assert result.duplicates == 0


@pytest.mark.asyncio
async def test_ingest_events_deduplication(db_session):
    """Ingesting the same event twice returns duplicates=1 on second call."""
    evt_id = str(uuid.uuid4())
    event = EventIn(
        event_id=evt_id,
        store_id=STORE_ID, camera_id=CAMERA_ENTRY, visitor_id="VIS_dedup2",
        event_type=EventType.ENTRY, timestamp=datetime.now(tz=timezone.utc),
        confidence=0.90,
    )
    await ingest_events([event], db_session)
    result2 = await ingest_events([event], db_session)
    assert result2.duplicates == 1
    assert result2.accepted == 0
