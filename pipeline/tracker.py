"""
StoreTracker: per-camera orchestration layer.

Each camera gets one StoreTracker instance. It processes YOLOv8 results frame
by frame and coordinates:
  - Zone mapping (which named zone is each person in?)
  - Entry / exit line crossing detection
  - Staff detection (colour + contextual)
  - Re-ID (same person across time / after re-entry)
  - Event emission (ENTRY, EXIT, ZONE_ENTER, ZONE_EXIT, ZONE_DWELL, BILLING_QUEUE_JOIN,
                    BILLING_QUEUE_ABANDON, REENTRY)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np

from pipeline.emit import EventEmitter, make_event
from pipeline.reid import ReIDManager
from pipeline.staff_detector import StaffDetector
from pipeline.zone_mapper import Zone, ZoneMapper

logger = logging.getLogger(__name__)

DWELL_EMIT_INTERVAL_S = 30   # emit ZONE_DWELL every 30 s of continued presence


@dataclass
class TrackState:
    """Mutable state for one active track."""
    track_id: int
    visitor_id: str
    is_staff: bool
    staff_confidence: float
    current_zone: Optional[Zone] = None
    zone_enter_time: Optional[datetime] = None
    last_dwell_emit_time: Optional[datetime] = None
    session_seq: int = 0
    prev_cx: Optional[float] = None
    prev_cy: Optional[float] = None
    entered_store: bool = False     # True once ENTRY event has been emitted
    queue_joined: bool = False      # True once BILLING_QUEUE_JOIN emitted


class StoreTracker:
    """
    Processes one camera's detections. Create one instance per clip.
    """

    def __init__(
        self,
        store_id: str,
        camera_id: str,
        zone_mapper: ZoneMapper,
        staff_detector: StaffDetector,
        reid_manager: ReIDManager,
        emitter: EventEmitter,
        fps: float,
        clip_start_time: datetime,
        current_billing_queue: "list[int]",  # shared mutable list across trackers
    ):
        self._store_id = store_id
        self._camera_id = camera_id
        self._zone_mapper = zone_mapper
        self._staff_detector = staff_detector
        self._reid = reid_manager
        self._emitter = emitter
        self._fps = fps
        self._clip_start = clip_start_time
        self._billing_queue = current_billing_queue  # track_ids currently in billing zone

        self._tracks: dict[int, TrackState] = {}
        self._seen_track_ids: set[int] = set()

    # ------------------------------------------------------------------
    # Main update — called every processed frame
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray, results, _frame_idx: int, frame_time_ms: float) -> None:
        """
        results: ultralytics Results object from model.track()
        frame_time_ms: elapsed milliseconds since clip start
        """
        now = self._clip_start + timedelta(milliseconds=frame_time_ms)
        active_ids: set[int] = set()

        boxes = results[0].boxes if results and results[0].boxes is not None else None
        if boxes is None or boxes.id is None:
            self._handle_lost_tracks(active_ids, now, frame)
            return

        ids = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for track_id, box, conf in zip(ids, xyxy, confs):
            track_id = int(track_id)
            active_ids.add(track_id)
            x1, y1, x2, y2 = box.tolist()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            bbox = (int(x1), int(y1), int(x2), int(y2))

            # First time seeing this track
            is_new = track_id not in self._tracks
            visitor_id, is_reentry = self._reid.register_or_match(
                track_id, frame, bbox, is_new
            )

            if is_new:
                is_staff, s_conf = self._staff_detector.is_staff(
                    track_id, frame, bbox, now
                )
                state = TrackState(
                    track_id=track_id,
                    visitor_id=visitor_id,
                    is_staff=is_staff,
                    staff_confidence=s_conf,
                    prev_cx=cx,
                    prev_cy=cy,
                )
                self._tracks[track_id] = state
                self._seen_track_ids.add(track_id)
            else:
                state = self._tracks[track_id]

            # Update staff zone history
            current_zone = self._zone_mapper.zone_at(cx, cy)
            if current_zone:
                self._staff_detector.update_zone(track_id, current_zone.zone_id, now)
                # Refine staff detection contextually once we have zone history
                if not state.is_staff:
                    ctx_staff = self._staff_detector.is_staff_by_context(track_id, now)
                    if ctx_staff:
                        state.is_staff = True
                        state.staff_confidence = 0.75

            # Entry / exit line crossing
            if self._zone_mapper.has_entry_line and state.prev_cx is not None:
                crossing = self._zone_mapper.check_entry_crossing(
                    state.prev_cx, state.prev_cy, cx, cy
                )
                if crossing == "ENTRY" and not state.entered_store:
                    state.entered_store = True
                    event_type = "REENTRY" if is_reentry else "ENTRY"
                    self._emit(event_type, state, now, conf, zone_id=None)
                elif crossing == "EXIT" and state.entered_store:
                    self._handle_exit(state, now, conf)

            # Zone transitions
            self._handle_zone_transition(state, current_zone, now, float(conf))

            # ZONE_DWELL — emit every 30 s of continued dwell
            self._handle_dwell(state, now, float(conf))

            state.prev_cx = cx
            state.prev_cy = cy

        # Tracks that disappeared this frame
        self._handle_lost_tracks(active_ids, now, frame)

    # ------------------------------------------------------------------
    # Event helpers
    # ------------------------------------------------------------------

    def _handle_zone_transition(
        self, state: TrackState, new_zone: Optional[Zone], now: datetime, conf: float
    ) -> None:
        if new_zone == state.current_zone:
            return

        # ZONE_EXIT from old zone
        if state.current_zone and state.entered_store:
            dwell_ms = 0
            if state.zone_enter_time:
                dwell_ms = int((now - state.zone_enter_time).total_seconds() * 1000)
            self._emit("ZONE_EXIT", state, now, conf,
                       zone_id=state.current_zone.zone_id, dwell_ms=dwell_ms)

            # Billing queue abandon: left billing without purchase
            if state.current_zone.is_billing and state.queue_joined:
                self._emit("BILLING_QUEUE_ABANDON", state, now, conf,
                           zone_id=state.current_zone.zone_id)
                state.queue_joined = False
                if state.track_id in self._billing_queue:
                    self._billing_queue.remove(state.track_id)

        state.current_zone = new_zone
        state.zone_enter_time = now
        state.last_dwell_emit_time = now

        # ZONE_ENTER into new zone
        if new_zone and state.entered_store:
            queue_depth = len(self._billing_queue)
            self._emit("ZONE_ENTER", state, now, conf,
                       zone_id=new_zone.zone_id, sku_zone=new_zone.sku_zone)

            if new_zone.is_billing:
                if queue_depth > 0:
                    self._emit("BILLING_QUEUE_JOIN", state, now, conf,
                               zone_id=new_zone.zone_id, queue_depth=queue_depth + 1)
                    state.queue_joined = True
                self._billing_queue.append(state.track_id)

    def _handle_dwell(self, state: TrackState, now: datetime, conf: float) -> None:
        if not state.current_zone or not state.zone_enter_time or not state.entered_store:
            return
        if state.last_dwell_emit_time is None:
            state.last_dwell_emit_time = state.zone_enter_time
        elapsed_since_emit = (now - state.last_dwell_emit_time).total_seconds()
        if elapsed_since_emit >= DWELL_EMIT_INTERVAL_S:
            self._emit(
                "ZONE_DWELL", state, now, conf,
                zone_id=state.current_zone.zone_id,
                dwell_ms=int(elapsed_since_emit * 1000),
                sku_zone=state.current_zone.sku_zone,
            )
            state.last_dwell_emit_time = now

    def _handle_exit(self, state: TrackState, now: datetime, conf: float) -> None:
        if state.current_zone:
            dwell_ms = 0
            if state.zone_enter_time:
                dwell_ms = int((now - state.zone_enter_time).total_seconds() * 1000)
            self._emit("ZONE_EXIT", state, now, conf,
                       zone_id=state.current_zone.zone_id, dwell_ms=dwell_ms)
            if state.current_zone.is_billing and state.queue_joined:
                self._emit("BILLING_QUEUE_ABANDON", state, now, conf,
                           zone_id=state.current_zone.zone_id)
                if state.track_id in self._billing_queue:
                    self._billing_queue.remove(state.track_id)
        self._emit("EXIT", state, now, conf, zone_id=None)
        self._reid.retire_track(state.track_id, last_zone=state.current_zone.zone_id if state.current_zone else None)
        del self._tracks[state.track_id]

    def _handle_lost_tracks(self, active_ids: set[int], now: datetime, _frame: np.ndarray) -> None:
        lost = [tid for tid in list(self._tracks) if tid not in active_ids]
        for tid in lost:
            state = self._tracks[tid]
            if state.entered_store:
                self._handle_exit(state, now, conf=state.staff_confidence or 0.5)
            else:
                self._reid.retire_track(tid)
                del self._tracks[tid]

    def flush_sessions(self) -> None:
        """Emit EXIT for all still-active tracks at end of clip."""
        now = datetime.now(tz=timezone.utc)
        for state in list(self._tracks.values()):
            if state.entered_store:
                self._handle_exit(state, now, conf=0.5)

    def _emit(
        self,
        event_type: str,
        state: TrackState,
        now: datetime,
        conf: float,
        zone_id: Optional[str] = None,
        dwell_ms: int = 0,
        sku_zone: Optional[str] = None,
        queue_depth: Optional[int] = None,
    ) -> None:
        state.session_seq += 1
        evt = make_event(
            store_id=self._store_id,
            camera_id=self._camera_id,
            visitor_id=state.visitor_id,
            event_type=event_type,
            timestamp=now,
            zone_id=zone_id,
            dwell_ms=dwell_ms,
            is_staff=state.is_staff,
            confidence=float(conf),
            queue_depth=queue_depth,
            sku_zone=sku_zone,
            session_seq=state.session_seq,
        )
        self._emitter.emit(evt)
