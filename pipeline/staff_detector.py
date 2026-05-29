"""
Staff detection via HSV colour analysis of the clothing region.

Primary method: Compare the dominant HSV colour in the lower-2/3 of the
bounding box against staff uniform colour ranges from store_layout.json.

Contextual heuristic (no image needed): if a track_id has traversed more
than 60% of distinct zones within a 3-minute window it is very likely staff.
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _hsv_in_range(hsv_pixel: np.ndarray, lower: list, upper: list) -> bool:
    """Check if HSV pixel falls in [lower, upper] (each a [H,S,V] list)."""
    lo = np.array(lower, dtype=np.uint8)
    hi = np.array(upper, dtype=np.uint8)
    return bool(np.all(hsv_pixel >= lo) and np.all(hsv_pixel <= hi))


class StaffDetector:
    """
    Per-camera staff detector. Call update() each frame.
    is_staff(track_id) returns True/False + confidence.
    """

    CONTEXTUAL_WINDOW_S = 180   # 3 minutes
    CONTEXTUAL_ZONE_THRESHOLD = 0.6

    def __init__(self, uniform_colors: Optional[list] = None, num_zones: int = 5):
        """
        uniform_colors: list of {"lower_hsv": [...], "upper_hsv": [...]} dicts.
        num_zones: total number of zones in this camera's coverage.
        """
        self._ranges: list[dict] = uniform_colors or []
        self._num_zones = max(num_zones, 1)
        # track_id → deque of (timestamp, zone_id) for contextual detection
        self._zone_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=200))
        # Cached decisions so we don't recompute every frame
        self._staff_cache: dict[int, bool] = {}

    @classmethod
    def from_layout(cls, layout_path: str, store_id: str) -> "StaffDetector":
        try:
            with open(layout_path, encoding="utf-8") as f:
                layout = json.load(f)
            stores = layout.get("stores", [layout])
            store_cfg = next((s for s in stores if s["store_id"] == store_id), {})
            colors_cfg = store_cfg.get("staff_uniform_colors", {})
            ranges = colors_cfg.get("ranges", [])
            # Also support legacy single-range format
            if not ranges and "primary_hsv" in colors_cfg:
                lower, upper = colors_cfg["primary_hsv"]
                ranges = [{"lower_hsv": lower, "upper_hsv": upper}]
            num_zones = len(store_cfg.get("zones", []))
            return cls(uniform_colors=ranges, num_zones=num_zones)
        except Exception as exc:
            logger.warning("staff_detector_layout_load_failed error=%s", exc)
            return cls()

    def update_zone(self, track_id: int, zone_id: str, timestamp: datetime) -> None:
        """Record zone visits for contextual detection."""
        self._zone_history[track_id].append((timestamp, zone_id))
        # Invalidate cache when new zone info arrives
        self._staff_cache.pop(track_id, None)

    def is_staff_by_color(self, frame: np.ndarray, bbox: tuple[int, int, int, int], track_id: int = -1) -> tuple[bool, float]:
        """
        Analyse the clothing region (lower 2/3 of bbox) in HSV space.
        Returns (is_staff, confidence).
        """
        if not self._ranges:
            return False, 0.0

        try:
            import cv2
            x1, y1, x2, y2 = bbox
            h = y2 - y1
            # Clothing region: lower 2/3 of the bounding box
            clothing_y1 = y1 + h // 3
            crop = frame[clothing_y1:y2, x1:x2]
            if crop.size == 0:
                return False, 0.0

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            # Find dominant colour using k-means (k=1 is just mean)
            pixels = hsv.reshape(-1, 3).astype(np.float32)
            mean_hsv = pixels.mean(axis=0).astype(np.uint8)

            for r in self._ranges:
                if _hsv_in_range(mean_hsv, r["lower_hsv"], r["upper_hsv"]):
                    # Confidence based on what fraction of pixels match
                    mask = cv2.inRange(hsv, np.array(r["lower_hsv"], dtype=np.uint8),
                                       np.array(r["upper_hsv"], dtype=np.uint8))
                    ratio = np.count_nonzero(mask) / mask.size
                    return True, round(float(ratio), 3)
            return False, 0.0
        except Exception as exc:
            logger.debug("staff_color_check_error track=%d error=%s", track_id, exc)
            return False, 0.0

    def is_staff_by_context(self, track_id: int, now: datetime) -> bool:
        """
        Contextual heuristic: high zone-traversal frequency within window → staff.
        Does not require image data.
        """
        if track_id in self._staff_cache:
            return self._staff_cache[track_id]

        history = self._zone_history.get(track_id)
        if not history:
            return False

        cutoff = now - timedelta(seconds=self.CONTEXTUAL_WINDOW_S)
        recent = [z for ts, z in history if ts >= cutoff]
        if len(recent) < 3:
            return False

        unique_zones = len(set(recent))
        ratio = unique_zones / self._num_zones
        is_staff = ratio >= self.CONTEXTUAL_ZONE_THRESHOLD
        self._staff_cache[track_id] = is_staff
        return is_staff

    def is_staff(
        self,
        track_id: int,
        frame: Optional[np.ndarray],
        bbox: Optional[tuple[int, int, int, int]],
        now: Optional[datetime] = None,
    ) -> tuple[bool, float]:
        """
        Combined check. Returns (is_staff, confidence).
        Context check only triggers if colour check is inconclusive.
        """
        if frame is not None and bbox is not None:
            by_color, conf = self.is_staff_by_color(frame, bbox)
            if by_color:
                return True, conf

        if now is not None and self.is_staff_by_context(track_id, now):
            return True, 0.75  # Heuristic confidence

        return False, 0.0
