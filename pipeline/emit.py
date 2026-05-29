"""
Event schema validation and emission.
EventEmitter batches events locally and flushes to the API (or a JSONL file).
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional
import requests

logger = logging.getLogger(__name__)

VALID_EVENT_TYPES = {
    "ENTRY", "EXIT", "ZONE_ENTER", "ZONE_EXIT",
    "ZONE_DWELL", "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON", "REENTRY",
}


@dataclass
class EventMetadata:
    queue_depth: Optional[int] = None
    sku_zone: Optional[str] = None
    session_seq: Optional[int] = None


@dataclass
class StoreEvent:
    event_id: str
    store_id: str
    camera_id: str
    visitor_id: str
    event_type: str
    timestamp: str          # ISO-8601 UTC
    zone_id: Optional[str]
    dwell_ms: int
    is_staff: bool
    confidence: float
    metadata: EventMetadata

    def to_dict(self) -> dict:
        d = asdict(self)
        d["metadata"] = asdict(self.metadata)
        return d

    def validate(self) -> None:
        assert self.event_type in VALID_EVENT_TYPES, f"Unknown event_type: {self.event_type}"
        assert 0.0 <= self.confidence <= 1.0, "confidence must be in [0, 1]"
        assert self.dwell_ms >= 0, "dwell_ms must be >= 0"
        zone_events = {"ZONE_ENTER", "ZONE_EXIT", "ZONE_DWELL",
                       "BILLING_QUEUE_JOIN", "BILLING_QUEUE_ABANDON"}
        if self.event_type in zone_events:
            assert self.zone_id, f"zone_id required for {self.event_type}"


def make_event(
    store_id: str,
    camera_id: str,
    visitor_id: str,
    event_type: str,
    timestamp: datetime,
    zone_id: Optional[str] = None,
    dwell_ms: int = 0,
    is_staff: bool = False,
    confidence: float = 1.0,
    queue_depth: Optional[int] = None,
    sku_zone: Optional[str] = None,
    session_seq: int = 0,
) -> StoreEvent:
    ts = timestamp.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    evt = StoreEvent(
        event_id=str(uuid.uuid4()),
        store_id=store_id,
        camera_id=camera_id,
        visitor_id=visitor_id,
        event_type=event_type,
        timestamp=ts,
        zone_id=zone_id,
        dwell_ms=dwell_ms,
        is_staff=is_staff,
        confidence=round(confidence, 4),
        metadata=EventMetadata(
            queue_depth=queue_depth,
            sku_zone=sku_zone,
            session_seq=session_seq,
        ),
    )
    evt.validate()
    return evt


class EventEmitter:
    """
    Buffers events in memory and flushes in batches to the API.
    Falls back to writing a JSONL file if the API is unreachable.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        batch_size: int = 100,
        output_file: Optional[str] = None,
    ):
        self._api_url = api_url.rstrip("/")
        self._batch_size = batch_size
        self._buffer: list[StoreEvent] = []
        self._output_file = output_file
        self._file_handle = None
        if output_file:
            self._file_handle = open(output_file, "a", encoding="utf-8")
        self._session = requests.Session()
        self._session.headers["Content-Type"] = "application/json"

    def emit(self, event: StoreEvent) -> None:
        self._buffer.append(event)
        if len(self._buffer) >= self._batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        batch = self._buffer[:]
        self._buffer.clear()

        if self._file_handle:
            for evt in batch:
                self._file_handle.write(json.dumps(evt.to_dict()) + "\n")
            self._file_handle.flush()

        try:
            payload = {"events": [evt.to_dict() for evt in batch]}
            resp = self._session.post(
                f"{self._api_url}/events/ingest",
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            result = resp.json()
            logger.info(
                "batch_flushed accepted=%d duplicates=%d rejected=%d",
                result.get("accepted", 0),
                result.get("duplicates", 0),
                result.get("rejected", 0),
            )
        except Exception as exc:
            logger.warning("api_flush_failed error=%s buffering_locally=%d", exc, len(batch))
            # Re-buffer if API is down; write to file as fallback
            if self._file_handle:
                for evt in batch:
                    self._file_handle.write(json.dumps(evt.to_dict()) + "\n")
                self._file_handle.flush()

    def close(self) -> None:
        self.flush()
        if self._file_handle:
            self._file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
