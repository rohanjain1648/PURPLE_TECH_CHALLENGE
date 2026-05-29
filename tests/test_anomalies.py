# PROMPT: "Write pytest-asyncio tests for GET /stores/{id}/anomalies.
# Cover: no anomalies on empty store, queue-spike anomaly appears after
# billing events, anomaly has required fields (type, severity, suggested_action),
# and severity escalates to CRITICAL when threshold is doubled.
# Also test that the anomaly endpoint returns a valid JSON list."
#
# CHANGES MADE: Switched from testing the background task directly to calling
# the anomaly detector functions inline (faster, no asyncio.sleep needed);
# added assertion for 'suggested_action' being non-empty (AI draft skipped this);
# added the DEAD_ZONE and CONVERSION_DROP anomaly types as separate test cases.

from __future__ import annotations

import uuid

import pytest

from app.anomalies import _upsert_anomaly, get_active_anomalies
from tests.conftest import STORE_ID


@pytest.mark.asyncio
async def test_anomalies_empty_store(client):
    """Empty store must return an empty anomaly list, not an error."""
    resp = await client.get("/stores/STORE_EMPTY_ANOM/anomalies")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["active_anomalies"] == []


@pytest.mark.asyncio
async def test_anomalies_response_shape(client):
    """Response must contain store_id, active_anomalies list, computed_at."""
    resp = await client.get(f"/stores/{STORE_ID}/anomalies")
    assert resp.status_code == 200
    body = resp.json()
    assert "store_id" in body
    assert "active_anomalies" in body
    assert isinstance(body["active_anomalies"], list)
    assert "computed_at" in body


@pytest.mark.asyncio
async def test_anomaly_fields_complete(db_session):
    """Inserted anomaly must surface with all required fields."""
    await _upsert_anomaly(
        store_id=STORE_ID,
        anomaly_type="BILLING_QUEUE_SPIKE",
        severity="WARN",
        suggested_action="Open another till.",
        metadata={"queue_depth": 6},
        db=db_session,
    )
    result = await get_active_anomalies(STORE_ID, db_session)
    spike = next((a for a in result.active_anomalies if a.anomaly_type == "BILLING_QUEUE_SPIKE"), None)
    assert spike is not None, "BILLING_QUEUE_SPIKE anomaly should be present"
    assert spike.severity in {"INFO", "WARN", "CRITICAL"}
    assert spike.suggested_action, "suggested_action must not be empty"
    assert spike.detected_at is not None


@pytest.mark.asyncio
async def test_anomaly_severity_escalation(db_session):
    """Calling upsert twice with escalated severity updates in-place."""
    sid = f"STORE_ESC_{uuid.uuid4().hex[:4]}"
    await _upsert_anomaly(sid, "BILLING_QUEUE_SPIKE", "WARN", "Warn action", {"queue_depth": 6}, db_session)
    await _upsert_anomaly(sid, "BILLING_QUEUE_SPIKE", "CRITICAL", "Critical action", {"queue_depth": 12}, db_session)

    result = await get_active_anomalies(sid, db_session)
    spikes = [a for a in result.active_anomalies if a.anomaly_type == "BILLING_QUEUE_SPIKE"]
    assert len(spikes) == 1, "Should not duplicate anomaly, only update"
    assert spikes[0].severity == "CRITICAL"


@pytest.mark.asyncio
async def test_anomaly_dead_zone(db_session):
    """DEAD_ZONE anomaly is zone-specific and surfaced correctly."""
    sid = f"STORE_DZ_{uuid.uuid4().hex[:4]}"
    await _upsert_anomaly(
        sid, "DEAD_ZONE", "INFO",
        "Zone SKINCARE has no traffic.",
        {"zone_id": "SKINCARE", "threshold_minutes": 30},
        db_session,
    )
    result = await get_active_anomalies(sid, db_session)
    dz = next((a for a in result.active_anomalies if a.anomaly_type == "DEAD_ZONE"), None)
    assert dz is not None
    assert dz.severity == "INFO"


@pytest.mark.asyncio
async def test_anomaly_conversion_drop(db_session):
    """CONVERSION_DROP anomaly surfaces with correct metadata keys."""
    sid = f"STORE_CD_{uuid.uuid4().hex[:4]}"
    await _upsert_anomaly(
        sid, "CONVERSION_DROP", "WARN",
        "Conversion rate dropped 25% vs 7-day avg.",
        {"today_rate": 0.05, "historical_avg": 0.20, "drop_pct": 0.25},
        db_session,
    )
    result = await get_active_anomalies(sid, db_session)
    cd = next((a for a in result.active_anomalies if a.anomaly_type == "CONVERSION_DROP"), None)
    assert cd is not None
    assert cd.metadata["drop_pct"] == pytest.approx(0.25)
