"""
Zone mapping: converts (x, y) pixel coordinates to named store zones.
Loads zone polygons from store_layout.json and uses Shapely for point-in-polygon tests.
Entry/exit detection is done via a virtual crossing line at the store threshold.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from shapely.geometry import Point, Polygon, LineString
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    logger.warning("shapely not installed — falling back to bounding-box zone detection")


@dataclass
class Zone:
    zone_id: str
    name: str
    camera_id: str
    sku_zone: Optional[str]
    polygon_pts: list[tuple[float, float]]
    is_billing: bool = False
    _polygon: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if SHAPELY_AVAILABLE and self.polygon_pts:
            self._polygon = Polygon(self.polygon_pts)

    def contains(self, x: float, y: float) -> bool:
        if SHAPELY_AVAILABLE and self._polygon:
            return self._polygon.contains(Point(x, y))
        # Fallback: axis-aligned bounding box
        xs = [p[0] for p in self.polygon_pts]
        ys = [p[1] for p in self.polygon_pts]
        return min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys)


@dataclass
class EntryLine:
    """Virtual line across the camera frame; crossing direction = ENTRY or EXIT."""
    p1: tuple[float, float]
    p2: tuple[float, float]
    # Which side of the line (positive/negative cross-product) is "inside the store"
    inside_sign: int = 1   # +1 or -1

    def side(self, x: float, y: float) -> int:
        """Returns +1 or -1 depending on which side of the line the point is."""
        dx = self.p2[0] - self.p1[0]
        dy = self.p2[1] - self.p1[1]
        cross = dx * (y - self.p1[1]) - dy * (x - self.p1[0])
        return 1 if cross >= 0 else -1

    def crossed(self, prev_x: float, prev_y: float, curr_x: float, curr_y: float) -> Optional[str]:
        """
        Returns 'ENTRY' if the centroid crossed from outside to inside,
        'EXIT' if inside to outside, None if no crossing.
        """
        prev_side = self.side(prev_x, prev_y)
        curr_side = self.side(curr_x, curr_y)
        if prev_side == curr_side:
            return None
        # curr_side == inside_sign means the person just moved inside
        if curr_side == self.inside_sign:
            return "ENTRY"
        return "EXIT"


class ZoneMapper:
    """
    Loaded once per store. Maps centroid coordinates → zone names.
    Also exposes the entry line for ENTRY/EXIT event detection.
    """

    def __init__(self, layout_path: str, store_id: str, camera_id: str):
        self._zones: list[Zone] = []
        self._entry_line: Optional[EntryLine] = None
        self._load(layout_path, store_id, camera_id)

    def _load(self, layout_path: str, store_id: str, camera_id: str) -> None:
        with open(layout_path, encoding="utf-8") as f:
            layout = json.load(f)

        # Support both {"stores": [...]} and single-store {"store_id": ...} formats
        stores = layout.get("stores", [layout])
        store_cfg = next((s for s in stores if s["store_id"] == store_id), None)
        if store_cfg is None:
            raise ValueError(f"Store {store_id} not found in {layout_path}")

        for z in store_cfg.get("zones", []):
            if z.get("camera_id") != camera_id:
                continue
            pts = [tuple(p) for p in z["polygon"]]
            zone = Zone(
                zone_id=z["zone_id"],
                name=z.get("name", z["zone_id"]),
                camera_id=camera_id,
                sku_zone=z.get("sku_zone"),
                polygon_pts=pts,
                is_billing=z.get("is_billing", False),
            )
            self._zones.append(zone)

            # Entry line (only on entry cameras)
            if "entry_line" in z:
                el = z["entry_line"]
                self._entry_line = EntryLine(
                    p1=tuple(el["p1"]),
                    p2=tuple(el["p2"]),
                    inside_sign=1,
                )

        logger.info(
            "zone_mapper_loaded store=%s camera=%s zones=%d has_entry_line=%s",
            store_id, camera_id, len(self._zones), self._entry_line is not None,
        )

    def zone_at(self, x: float, y: float) -> Optional[Zone]:
        """Return the first zone that contains (x, y), or None."""
        for zone in self._zones:
            if zone.contains(x, y):
                return zone
        return None

    def check_entry_crossing(
        self, prev_x: float, prev_y: float, curr_x: float, curr_y: float
    ) -> Optional[str]:
        """Returns 'ENTRY', 'EXIT', or None."""
        if self._entry_line is None:
            return None
        return self._entry_line.crossed(prev_x, prev_y, curr_x, curr_y)

    @property
    def has_entry_line(self) -> bool:
        return self._entry_line is not None
