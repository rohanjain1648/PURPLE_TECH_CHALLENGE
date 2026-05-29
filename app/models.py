"""
SQLAlchemy ORM models + Pydantic request/response schemas.
Single source of truth for every data shape in the system.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import (
    BigInteger,
    Boolean,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import DateTime, JSON


# ---------------------------------------------------------------------------
# SQLAlchemy base + ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class EventORM(Base):
    __tablename__ = "events"

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    store_id: Mapped[str] = mapped_column(String(64), nullable=False)
    camera_id: Mapped[str] = mapped_column(String(64), nullable=False)
    visitor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    zone_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    dwell_ms: Mapped[int] = mapped_column(Integer, default=0)
    is_staff: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    event_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSON, nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    __table_args__ = (
        Index("idx_events_store_ts", "store_id", "timestamp"),
        Index("idx_events_visitor", "visitor_id", "store_id"),
        Index("idx_events_type_store", "store_id", "event_type", "timestamp"),
    )


class SessionORM(Base):
    """One row per visitor session (re-entries create new rows)."""
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    visitor_id: Mapped[str] = mapped_column(String(64), nullable=False)
    store_id: Mapped[str] = mapped_column(String(64), nullable=False)
    entry_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    exit_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    zones_visited: Mapped[Optional[dict]] = mapped_column(JSON, default=list)
    billing_entry_time: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    converted: Mapped[bool] = mapped_column(Boolean, default=False)
    is_reentry: Mapped[bool] = mapped_column(Boolean, default=False)
    total_dwell_ms: Mapped[int] = mapped_column(BigInteger, default=0)
    queue_joined: Mapped[bool] = mapped_column(Boolean, default=False)
    queue_abandoned: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (
        Index("idx_sessions_store_entry", "store_id", "entry_time"),
        Index("idx_sessions_visitor_store", "visitor_id", "store_id"),
    )


class PosTransactionORM(Base):
    __tablename__ = "pos_transactions"

    transaction_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    store_id: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    basket_value_inr: Mapped[float] = mapped_column(Float, nullable=False)

    __table_args__ = (
        Index("idx_pos_store_ts", "store_id", "timestamp"),
    )


class AnomalyORM(Base):
    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    store_id: Mapped[str] = mapped_column(String(64), nullable=False)
    anomaly_type: Mapped[str] = mapped_column(String(64), nullable=False)
    severity: Mapped[str] = mapped_column(String(16), nullable=False)  # INFO/WARN/CRITICAL
    detected_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    suggested_action: Mapped[Optional[str]] = mapped_column(Text)
    anomaly_metadata: Mapped[Optional[dict]] = mapped_column("metadata", JSON)

    __table_args__ = (
        Index("idx_anomalies_store_active", "store_id", "resolved_at"),
    )


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    ZONE_ENTER = "ZONE_ENTER"
    ZONE_EXIT = "ZONE_EXIT"
    ZONE_DWELL = "ZONE_DWELL"
    BILLING_QUEUE_JOIN = "BILLING_QUEUE_JOIN"
    BILLING_QUEUE_ABANDON = "BILLING_QUEUE_ABANDON"
    REENTRY = "REENTRY"


class EventMetadata(BaseModel):
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None


class EventIn(BaseModel):
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: EventType
    timestamp: datetime
    zone_id: Optional[str] = None
    dwell_ms: int = 0
    is_staff: bool = False
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Optional[EventMetadata] = None

    @field_validator("dwell_ms")
    @classmethod
    def dwell_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("dwell_ms must be >= 0")
        return v

    @model_validator(mode="after")
    def zone_required_for_zone_events(self) -> "EventIn":
        zone_events = {EventType.ZONE_ENTER, EventType.ZONE_EXIT, EventType.ZONE_DWELL,
                       EventType.BILLING_QUEUE_JOIN, EventType.BILLING_QUEUE_ABANDON}
        if self.event_type in zone_events and not self.zone_id:
            raise ValueError(f"zone_id is required for event_type={self.event_type}")
        return self


class EventBatch(BaseModel):
    events: list[EventIn] = Field(min_length=1)

    @field_validator("events")
    @classmethod
    def max_batch_size(cls, v: list[EventIn]) -> list[EventIn]:
        if len(v) > 500:
            raise ValueError("Batch size exceeds maximum of 500 events")
        return v


class IngestResult(BaseModel):
    accepted: int
    duplicates: int
    rejected: int
    errors: list[dict[str, Any]] = Field(default_factory=list)


class ZoneDwellMetric(BaseModel):
    zone_id: str
    avg_dwell_ms: float
    visitor_count: int


class StoreMetrics(BaseModel):
    store_id: str
    date: str
    unique_visitors: int
    conversion_rate: float
    avg_dwell_per_zone: list[ZoneDwellMetric]
    current_queue_depth: int
    abandonment_rate: float
    active_visitors: int
    computed_at: datetime


class FunnelStage(BaseModel):
    stage: str
    count: int
    drop_off_pct: float


class FunnelResponse(BaseModel):
    store_id: str
    date: str
    stages: list[FunnelStage]
    computed_at: datetime


class HeatmapZone(BaseModel):
    zone_id: str
    zone_name: Optional[str] = None
    visit_count: int
    avg_dwell_ms: float
    normalized_score: float = Field(ge=0.0, le=100.0)
    data_confidence: str  # "HIGH" or "LOW"


class HeatmapResponse(BaseModel):
    store_id: str
    date: str
    zones: list[HeatmapZone]
    computed_at: datetime


class AnomalyResponse(BaseModel):
    id: int
    store_id: str
    anomaly_type: str
    severity: str
    detected_at: datetime
    suggested_action: str
    metadata: Optional[dict] = None


class AnomaliesResponse(BaseModel):
    store_id: str
    active_anomalies: list[AnomalyResponse]
    computed_at: datetime


class StoreHealth(BaseModel):
    store_id: str
    last_event_at: Optional[datetime]
    status: str  # HEALTHY / STALE_FEED / NO_DATA


class HealthResponse(BaseModel):
    status: str  # healthy / degraded / unhealthy
    database: str  # connected / disconnected
    stores: list[StoreHealth]
    uptime_seconds: float
    version: str
    checked_at: datetime


class PosTransactionIn(BaseModel):
    transaction_id: str
    store_id: str
    timestamp: datetime
    basket_value_inr: float = Field(gt=0)


class PosLoadRequest(BaseModel):
    transactions: list[PosTransactionIn]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[dict] = None
