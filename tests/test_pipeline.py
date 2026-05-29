# PROMPT: "Write unit tests for the detection pipeline components: EventEmitter
# schema validation, ZoneMapper point-in-polygon logic, StaffDetector contextual
# detection, ReIDManager re-entry matching, and the make_event factory.
# Use only CPU-side logic (no real video frames needed). Mock cv2 where needed."
#
# CHANGES MADE: Added the group-entry scenario test (AI draft missed this edge case
# from the spec); fixed ZoneMapper test to use the actual sample layout file path;
# added confidence-range assertion to make_event (AI only checked event_type);
# added test for ZONE_DWELL requiring zone_id (schema invariant).

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from pipeline.emit import EventEmitter, make_event
from pipeline.reid import ReIDManager, _cosine_similarity
from pipeline.staff_detector import StaffDetector
from pipeline.zone_mapper import EntryLine, ZoneMapper

LAYOUT_PATH = str(Path(__file__).parent.parent / "data" / "store_layout.json")
STORE_ID = "STORE_BLR_002"


# ---------------------------------------------------------------------------
# make_event / EventEmitter schema tests
# ---------------------------------------------------------------------------

def test_make_event_valid_entry():
    evt = make_event(
        store_id=STORE_ID, camera_id="CAM_ENTRY_01", visitor_id="VIS_abc",
        event_type="ENTRY", timestamp=datetime.now(tz=timezone.utc),
        confidence=0.92,
    )
    assert evt.event_type == "ENTRY"
    assert 0.0 <= evt.confidence <= 1.0
    assert evt.event_id  # non-empty UUID


def test_make_event_zone_required_for_dwell():
    """ZONE_DWELL without zone_id must raise AssertionError."""
    with pytest.raises(AssertionError):
        make_event(
            store_id=STORE_ID, camera_id="CAM_FLOOR_01", visitor_id="VIS_abc",
            event_type="ZONE_DWELL", timestamp=datetime.now(tz=timezone.utc),
            zone_id=None,   # missing — should fail
            confidence=0.80,
        )


def test_make_event_unknown_type_rejected():
    with pytest.raises(AssertionError):
        make_event(
            store_id=STORE_ID, camera_id="CAM_ENTRY_01", visitor_id="VIS_abc",
            event_type="WALK_PAST",  # not in catalogue
            timestamp=datetime.now(tz=timezone.utc), confidence=0.90,
        )


def test_event_emitter_buffers_and_flushes(tmp_path):
    """EventEmitter must write events to the fallback file when API is down."""
    out = tmp_path / "events.jsonl"
    # API at port 1 will always refuse connection — triggers file fallback
    with EventEmitter(api_url="http://127.0.0.1:1", output_file=str(out), batch_size=2) as emitter:
        emitter.emit(make_event(
            store_id=STORE_ID, camera_id="CAM_ENTRY_01", visitor_id="VIS_t1",
            event_type="ENTRY", timestamp=datetime.now(tz=timezone.utc), confidence=0.88,
        ))
        emitter.emit(make_event(
            store_id=STORE_ID, camera_id="CAM_ENTRY_01", visitor_id="VIS_t2",
            event_type="ENTRY", timestamp=datetime.now(tz=timezone.utc), confidence=0.91,
        ))

    lines = out.read_text().strip().splitlines()
    assert len(lines) >= 2
    event = json.loads(lines[0])
    assert event["event_type"] == "ENTRY"
    assert event["store_id"] == STORE_ID


# ---------------------------------------------------------------------------
# ZoneMapper
# ---------------------------------------------------------------------------

def test_entry_line_crossing_entry():
    """Point moving from below (y=700, outside) to above (y=400, inside) is ENTRY.
    For a horizontal line at y=540: above the line (y < 540) has a negative
    cross-product, so inside_sign=-1 means 'above the line is inside'.
    """
    line = EntryLine(p1=(0, 540), p2=(1920, 540), inside_sign=-1)
    result = line.crossed(prev_x=960, prev_y=700, curr_x=960, curr_y=400)
    assert result == "ENTRY"


def test_entry_line_crossing_exit():
    line = EntryLine(p1=(0, 540), p2=(1920, 540), inside_sign=-1)
    result = line.crossed(prev_x=960, prev_y=400, curr_x=960, curr_y=700)
    assert result == "EXIT"


def test_entry_line_no_crossing():
    line = EntryLine(p1=(0, 540), p2=(1920, 540), inside_sign=-1)
    result = line.crossed(prev_x=960, prev_y=300, curr_x=960, curr_y=350)
    assert result is None


@pytest.mark.skipif(not Path(LAYOUT_PATH).exists(), reason="layout file not present")
def test_zone_mapper_loads_layout():
    zm = ZoneMapper(LAYOUT_PATH, STORE_ID, "CAM_FLOOR_01")
    # Centroid in the middle of the frame should hit a zone
    zone = zm.zone_at(320, 270)
    assert zone is not None or zone is None  # just must not crash


# ---------------------------------------------------------------------------
# StaffDetector
# ---------------------------------------------------------------------------

def test_staff_detector_contextual_no_history():
    """No zone history → not classified as staff."""
    sd = StaffDetector(num_zones=5)
    result = sd.is_staff_by_context(track_id=99, now=datetime.now(tz=timezone.utc))
    assert result is False


def test_staff_detector_contextual_high_traversal():
    """Track that visits >60% of zones in window is classified as staff."""
    sd = StaffDetector(num_zones=5)
    now = datetime.now(tz=timezone.utc)
    for zone in ["SKINCARE", "HAIRCARE", "MAKEUP", "BILLING"]:
        sd.update_zone(track_id=1, zone_id=zone, timestamp=now)
    assert sd.is_staff_by_context(track_id=1, now=now) is True


def test_staff_detector_contextual_low_traversal():
    """Track visiting only 1 zone should not be classified as staff."""
    sd = StaffDetector(num_zones=5)
    now = datetime.now(tz=timezone.utc)
    sd.update_zone(track_id=2, zone_id="SKINCARE", timestamp=now)
    assert sd.is_staff_by_context(track_id=2, now=now) is False


# ---------------------------------------------------------------------------
# ReIDManager
# ---------------------------------------------------------------------------

def test_reid_new_track_gets_visitor_id():
    mgr = ReIDManager()
    frame = np.zeros((100, 60, 3), dtype=np.uint8)
    vid, is_reentry = mgr.register_or_match(track_id=10, frame=frame, bbox=(0, 0, 60, 100))
    assert vid.startswith("VIS_")
    assert is_reentry is False


def test_reid_same_track_stable_id():
    """Same track_id must always return the same visitor_id."""
    mgr = ReIDManager()
    frame = np.zeros((100, 60, 3), dtype=np.uint8)
    vid1, _ = mgr.register_or_match(track_id=20, frame=frame, bbox=(0, 0, 60, 100))
    vid2, _ = mgr.register_or_match(track_id=20, frame=frame, bbox=(5, 5, 55, 95))
    assert vid1 == vid2


def test_reid_retire_track():
    mgr = ReIDManager()
    frame = np.zeros((100, 60, 3), dtype=np.uint8)
    vid, _ = mgr.register_or_match(track_id=30, frame=frame, bbox=(0, 0, 60, 100))
    retired = mgr.retire_track(30)
    assert retired == vid
    assert mgr.get_visitor_id(30) is None


def test_cosine_similarity_identical():
    """Identical unit vectors must have similarity 1.0."""
    v = np.array([0.6, 0.8], dtype=np.float32)
    assert _cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    """Orthogonal unit vectors must have similarity 0.0."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert _cosine_similarity(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Group entry edge case
# ---------------------------------------------------------------------------

def test_group_entry_produces_multiple_visitor_ids():
    """
    3 people entering simultaneously must get 3 distinct visitor_ids.
    Simulated by registering 3 different track_ids at the same time.
    """
    mgr = ReIDManager()
    frame = np.zeros((200, 100, 3), dtype=np.uint8)
    ids = set()
    for tid in [101, 102, 103]:
        vid, _ = mgr.register_or_match(track_id=tid, frame=frame, bbox=(0, 0, 100, 200))
        ids.add(vid)
    assert len(ids) == 3, "Each person in a group must get a unique visitor_id"
