"""
Re-ID manager for cross-camera deduplication and re-entry detection.

Approach:
- Extract a compact appearance descriptor (normalised histogram in LAB space)
  from each tracked bounding box.
- When a track disappears, store its descriptor + timestamp in a short-term gallery.
- When a new track appears, compare its descriptor to the gallery via cosine similarity.
- If similarity > threshold AND time gap is within the re-entry window → REENTRY.
- For cross-camera matching, the same gallery is shared across all cameras of a store.
"""
from __future__ import annotations

import hashlib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.82   # cosine similarity above which tracks are the same person
GALLERY_TTL_S = 300           # 5 minutes — keep exited descriptors this long


@dataclass
class GalleryEntry:
    visitor_id: str
    descriptor: np.ndarray
    exited_at: float          # monotonic time
    last_zone: Optional[str] = None


def _extract_descriptor(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    32-bin LAB histogram over the bounding box (fast, lighting-invariant).
    Returns a unit-norm vector of shape (96,).
    """
    try:
        import cv2
        x1, y1, x2, y2 = bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        hists = []
        for ch in range(3):
            h = cv2.calcHist([lab], [ch], None, [32], [0, 256])
            hists.append(h.flatten())
        descriptor = np.concatenate(hists).astype(np.float32)
        norm = np.linalg.norm(descriptor)
        if norm == 0:
            return None
        return descriptor / norm
    except Exception as exc:
        logger.debug("descriptor_extraction_failed error=%s", exc)
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))  # both are already unit-norm


def _short_visitor_id(seed: str) -> str:
    """Generate a short, stable visitor ID from a seed string."""
    h = hashlib.sha256(seed.encode()).hexdigest()[:6]
    return f"VIS_{h}"


class ReIDManager:
    """
    Maintains an appearance gallery shared across cameras for one store.
    Thread-unsafe — call from a single pipeline process.
    """

    def __init__(self, reentry_window_s: int = 300):
        self._gallery: OrderedDict[str, GalleryEntry] = OrderedDict()
        self._reentry_window_s = reentry_window_s
        # active_tracks: track_id (int) → visitor_id (str)
        self._active: dict[int, str] = {}
        # descriptor cache per active track (updated periodically)
        self._track_descriptors: dict[int, np.ndarray] = {}
        self._track_counters: dict[int, int] = {}  # increments each time we see a track

    # ------------------------------------------------------------------
    # Called every frame for each detected track
    # ------------------------------------------------------------------

    def register_or_match(
        self,
        track_id: int,
        frame: np.ndarray,
        bbox: tuple[int, int, int, int],
        _is_new_track: bool = False,
    ) -> tuple[str, bool]:
        """
        Returns (visitor_id, is_reentry).
        is_reentry=True means this visitor was seen before (crossed exit line then re-appeared).
        """
        self._track_counters[track_id] = self._track_counters.get(track_id, 0) + 1

        if track_id in self._active:
            # Update descriptor periodically (every 15 frames)
            if self._track_counters[track_id] % 15 == 0:
                desc = _extract_descriptor(frame, bbox)
                if desc is not None:
                    self._track_descriptors[track_id] = desc
            return self._active[track_id], False

        # New track — try to match against gallery
        desc = _extract_descriptor(frame, bbox)
        if desc is not None:
            self._track_descriptors[track_id] = desc
            match = self._find_gallery_match(desc)
            if match:
                visitor_id = match.visitor_id
                # Remove matched entry so it isn't matched again
                self._gallery.pop(visitor_id, None)
                self._active[track_id] = visitor_id
                logger.debug("reentry_detected visitor=%s track=%d", visitor_id, track_id)
                return visitor_id, True

        # Brand-new visitor
        visitor_id = _short_visitor_id(f"{track_id}_{time.monotonic()}")
        self._active[track_id] = visitor_id
        return visitor_id, False

    # ------------------------------------------------------------------
    # Called when a track is lost (disappeared from frame or crossed exit)
    # ------------------------------------------------------------------

    def retire_track(self, track_id: int, last_zone: Optional[str] = None) -> Optional[str]:
        """Move track from active to gallery. Returns the visitor_id."""
        visitor_id = self._active.pop(track_id, None)
        if visitor_id is None:
            return None
        desc = self._track_descriptors.pop(track_id, None)
        self._track_counters.pop(track_id, None)
        if desc is not None:
            self._gallery[visitor_id] = GalleryEntry(
                visitor_id=visitor_id,
                descriptor=desc,
                exited_at=time.monotonic(),
                last_zone=last_zone,
            )
        self._evict_expired()
        return visitor_id

    def get_visitor_id(self, track_id: int) -> Optional[str]:
        return self._active.get(track_id)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _find_gallery_match(self, descriptor: np.ndarray) -> Optional[GalleryEntry]:
        now = time.monotonic()
        best_score = SIMILARITY_THRESHOLD
        best_entry: Optional[GalleryEntry] = None
        for entry in self._gallery.values():
            age_s = now - entry.exited_at
            if age_s > self._reentry_window_s:
                continue
            sim = _cosine_similarity(descriptor, entry.descriptor)
            if sim > best_score:
                best_score = sim
                best_entry = entry
        return best_entry

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, v in self._gallery.items()
                   if now - v.exited_at > GALLERY_TTL_S]
        for k in expired:
            del self._gallery[k]
