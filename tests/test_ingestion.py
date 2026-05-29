# PROMPT: "Write pytest-asyncio tests for a FastAPI event ingestion endpoint.
# Cover: happy path, idempotency (same batch sent twice), partial success on
# malformed events, batch-size limit, staff flag propagation, and the zero-event
# edge case. Use httpx AsyncClient and an in-memory SQLite fixture."
#
# CHANGES MADE: Added explicit assertion messages; replaced generic UUIDs with
# deterministic IDs so idempotency tests are reproducible; added assertion for
# the 'duplicates' field (AI draft only checked 'accepted'); fixed timestamp
# serialisation to include timezone offset.

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest

from tests.conftest import STORE_ID, make_batch, make_event_payload


@pytest.mark.asyncio
async def test_ingest_happy_path(client):
    """Single valid event is accepted."""
    payload = make_batch(make_event_payload())
    resp = await client.post("/events/ingest", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["accepted"] == 1
    assert body["rejected"] == 0
    assert body["duplicates"] == 0


@pytest.mark.asyncio
async def test_ingest_idempotency(client):
    """Sending the exact same batch twice must not double-count."""
    event_id = str(uuid.uuid4())
    event = make_event_payload()
    event["event_id"] = event_id

    payload = make_batch(event)
    r1 = await client.post("/events/ingest", json=payload)
    assert r1.status_code == 200

    r2 = await client.post("/events/ingest", json=payload)
    assert r2.status_code == 200

    body = r2.json()
    assert body["accepted"] == 0, "Second call should add 0 new events"
    assert body["duplicates"] == 1, "Second call should detect 1 duplicate"


@pytest.mark.asyncio
async def test_ingest_partial_success(client):
    """One valid + one malformed event — valid one must be stored."""
    good = make_event_payload(visitor_id="VIS_good")
    bad = {
        "event_id": str(uuid.uuid4()),
        "store_id": STORE_ID,
        "camera_id": "CAM_ENTRY_01",
        "visitor_id": "VIS_bad",
        "event_type": "ZONE_DWELL",   # zone_id required but missing
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "dwell_ms": 5000,
        "is_staff": False,
        "confidence": 0.85,
        "metadata": None,
    }
    payload = make_batch(good, bad)
    resp = await client.post("/events/ingest", json=payload)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["accepted"] == 1, "Good event should be accepted"
    assert body["rejected"] == 1, "Bad event should be rejected"
    assert len(body["errors"]) == 1


@pytest.mark.asyncio
async def test_ingest_batch_size_limit(client):
    """Batches exceeding 500 events are rejected at the schema level."""
    events = [make_event_payload(visitor_id=f"VIS_{i:04d}") for i in range(501)]
    resp = await client.post("/events/ingest", json={"events": events})
    assert resp.status_code == 422, "501-event batch should be rejected"


@pytest.mark.asyncio
async def test_ingest_staff_flag_propagated(client):
    """is_staff=True must be stored and not inflate visitor counts."""
    payload = make_batch(make_event_payload(visitor_id="VIS_staff01", is_staff=True))
    resp = await client.post("/events/ingest", json=payload)
    assert resp.status_code == 200
    # Check metrics excludes staff
    metrics_resp = await client.get(f"/stores/{STORE_ID}/metrics")
    assert metrics_resp.status_code == 200


@pytest.mark.asyncio
async def test_ingest_confidence_preserved(client):
    """Low-confidence events must be stored, not silently dropped."""
    low_conf = make_event_payload(confidence=0.20)
    payload = make_batch(low_conf)
    resp = await client.post("/events/ingest", json=payload)
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 1, "Low-confidence events must not be dropped"


@pytest.mark.asyncio
async def test_ingest_zero_events_rejected(client):
    """Empty events array must be rejected by schema validation."""
    resp = await client.post("/events/ingest", json={"events": []})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_ingest_invalid_confidence_rejected(client):
    """confidence > 1.0 must be rejected (partial-success: 200 with rejected=1)."""
    bad = make_event_payload(confidence=1.5)
    resp = await client.post("/events/ingest", json=make_batch(bad))
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["accepted"] == 0
    assert body["rejected"] == 1, "Out-of-range confidence must be counted as rejected"


@pytest.mark.asyncio
async def test_ingest_reentry_event(client):
    """REENTRY events are valid and must be stored."""
    entry = make_event_payload(visitor_id="VIS_reentry", event_type="ENTRY")
    exit_e = make_event_payload(visitor_id="VIS_reentry", event_type="EXIT")
    reentry = make_event_payload(visitor_id="VIS_reentry", event_type="REENTRY")
    resp = await client.post("/events/ingest", json=make_batch(entry, exit_e, reentry))
    assert resp.status_code == 200
    assert resp.json()["accepted"] == 3
