"""
Shared pytest fixtures — in-memory SQLite DB, async client, seeded events.
"""
from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import get_db
from app.main import app
from app.models import Base, EventIn, EventType


# ---------------------------------------------------------------------------
# In-memory SQLite engine (per test session)
# ---------------------------------------------------------------------------

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

@pytest.fixture(scope="session")
def event_loop():
    """Single event loop for the whole test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    engine = create_async_engine(TEST_DB_URL, connect_args={"check_same_thread": False})
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(db_engine) -> AsyncGenerator[AsyncClient, None]:
    """AsyncClient that uses the test DB instead of the real one."""
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False, class_=AsyncSession)

    async def override_get_db():
        async with factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Event factories
# ---------------------------------------------------------------------------

STORE_ID = "STORE_BLR_002"
CAMERA_ENTRY = "CAM_ENTRY_01"
CAMERA_FLOOR = "CAM_FLOOR_01"
CAMERA_BILLING = "CAM_BILLING_01"


def make_event_payload(
    visitor_id: str = "VIS_abc123",
    event_type: str = "ENTRY",
    zone_id: str | None = None,
    is_staff: bool = False,
    dwell_ms: int = 0,
    confidence: float = 0.90,
    timestamp: datetime | None = None,
    camera_id: str = CAMERA_ENTRY,
    queue_depth: int | None = None,
) -> dict:
    ts = timestamp or datetime.now(tz=timezone.utc)
    return {
        "event_id": str(uuid.uuid4()),
        "store_id": STORE_ID,
        "camera_id": camera_id,
        "visitor_id": visitor_id,
        "event_type": event_type,
        "timestamp": ts.isoformat(),
        "zone_id": zone_id,
        "dwell_ms": dwell_ms,
        "is_staff": is_staff,
        "confidence": confidence,
        "metadata": {"queue_depth": queue_depth, "sku_zone": None, "session_seq": 1},
    }


def make_batch(*events: dict) -> dict:
    return {"events": list(events)}
