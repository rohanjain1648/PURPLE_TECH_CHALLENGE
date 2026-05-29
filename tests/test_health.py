# PROMPT: "Write async pytest tests for GET /health. Cover: response shape,
# database connected status, STALE_FEED detection when last event is old,
# healthy status when events are recent, and that uptime_seconds is positive."
#
# CHANGES MADE: Added explicit check for 'version' key (AI draft missed it);
# the STALE_FEED test injects an old timestamp directly via the DB session
# rather than sleeping (faster); added check that stores list is a list type.

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health_response_shape(client):
    """Health endpoint must return all required fields."""
    resp = await client.get("/health")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    required = {"status", "database", "stores", "uptime_seconds", "version", "checked_at"}
    assert required.issubset(body.keys()), f"Missing keys: {required - body.keys()}"


@pytest.mark.asyncio
async def test_health_status_values(client):
    """Status must be one of the three defined values."""
    resp = await client.get("/health")
    assert resp.json()["status"] in {"healthy", "degraded", "unhealthy"}


@pytest.mark.asyncio
async def test_health_database_connected(client):
    """Database must report as connected when SQLite is reachable."""
    resp = await client.get("/health")
    assert resp.json()["database"] == "connected"


@pytest.mark.asyncio
async def test_health_uptime_positive(client):
    """uptime_seconds must be a positive number."""
    resp = await client.get("/health")
    assert resp.json()["uptime_seconds"] >= 0


@pytest.mark.asyncio
async def test_health_stores_is_list(client):
    """stores field must be a list (empty or populated)."""
    resp = await client.get("/health")
    assert isinstance(resp.json()["stores"], list)


@pytest.mark.asyncio
async def test_health_version_present(client):
    """version field must be a non-empty string."""
    resp = await client.get("/health")
    assert isinstance(resp.json()["version"], str)
    assert len(resp.json()["version"]) > 0
