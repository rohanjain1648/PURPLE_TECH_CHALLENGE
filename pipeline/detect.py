"""
Main detection + tracking script.

Usage:
    python -m pipeline.detect \\
        --clips data/clips/ \\
        --layout data/store_layout.json \\
        --store-id STORE_BLR_002 \\
        --api-url http://localhost:8000 \\
        --output events.jsonl

Processes every clip found in --clips, matching filenames to camera IDs via
the naming convention:  <store_id>_<camera_id>_<anything>.mp4

Model: YOLOv8s with ByteTrack (via ultralytics' built-in tracker).
Frame skip: process every 3rd frame (15fps → effective 5fps) for CPU-feasibility;
override with --frame-skip 1 for full-rate GPU processing.
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

from pipeline.emit import EventEmitter
from pipeline.reid import ReIDManager
from pipeline.staff_detector import StaffDetector
from pipeline.tracker import StoreTracker
from pipeline.zone_mapper import ZoneMapper

logger = logging.getLogger(__name__)

# Camera ID derived from filename:  <store>_CAM_ENTRY_01_<...>.mp4 → CAM_ENTRY_01
_CAM_RE = re.compile(r"(CAM_[A-Z0-9_]+)", re.IGNORECASE)


def _parse_camera_id(filename: str) -> str:
    m = _CAM_RE.search(filename)
    return m.group(1).upper() if m else Path(filename).stem.upper()


def _parse_clip_start_time(clip_path: Path) -> datetime:
    """
    Attempt to read clip start time from filename (ISO format) or fall back to now.
    Expected pattern: ..._2026-03-03T14-00-00_...mp4
    """
    stem = clip_path.stem
    ts_re = re.search(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})", stem)
    if ts_re:
        try:
            return datetime.fromisoformat(ts_re.group(1).replace("-", ":", 2).replace("T", "T", 1)).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return datetime.now(tz=timezone.utc)


def _load_model(model_name: str = "yolov8s.pt"):
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        logger.info("yolo_model_loaded model=%s", model_name)
        return model
    except ImportError:
        logger.error("ultralytics not installed — pip install ultralytics")
        sys.exit(1)


def process_clip(
    clip_path: Path,
    store_id: str,
    camera_id: str,
    layout_path: str,
    emitter: EventEmitter,
    model,
    reid_manager: ReIDManager,
    billing_queue: list,
    confidence_threshold: float = 0.40,
    frame_skip: int = 3,
) -> None:
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.error("cannot_open_clip path=%s", clip_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    clip_start = _parse_clip_start_time(clip_path)

    zone_mapper = ZoneMapper(layout_path, store_id, camera_id)
    staff_detector = StaffDetector.from_layout(layout_path, store_id)

    tracker = StoreTracker(
        store_id=store_id,
        camera_id=camera_id,
        zone_mapper=zone_mapper,
        staff_detector=staff_detector,
        reid_manager=reid_manager,
        emitter=emitter,
        fps=fps,
        clip_start_time=clip_start,
        current_billing_queue=billing_queue,
    )

    logger.info(
        "processing_clip path=%s camera=%s frames=%d fps=%.1f",
        clip_path.name, camera_id, total_frames, fps,
    )

    frame_idx = 0
    processed = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            frame_time_ms = (frame_idx / fps) * 1000.0
            results = model.track(
                frame,
                persist=True,
                classes=[0],           # person class only
                conf=confidence_threshold,
                iou=0.45,
                tracker="bytetrack.yaml",
                verbose=False,
            )
            tracker.update(frame, results, frame_idx, frame_time_ms)
            processed += 1

        frame_idx += 1

    tracker.flush_sessions()
    cap.release()
    logger.info("clip_done camera=%s total_frames=%d processed=%d", camera_id, frame_idx, processed)


def run(args: argparse.Namespace) -> None:
    clips_dir = Path(args.clips)
    layout_path = args.layout

    with EventEmitter(api_url=args.api_url, output_file=args.output) as emitter:
        model = _load_model(args.model)
        reid_manager = ReIDManager(reentry_window_s=300)
        billing_queue: list[int] = []

        # Process clips in order: ENTRY first, then FLOOR, then BILLING
        priority = {"ENTRY": 0, "FLOOR": 1, "BILLING": 2}
        clip_paths = sorted(
            clips_dir.glob("*.mp4"),
            key=lambda p: priority.get(next((k for k in priority if k in p.name.upper()), ""), 3),
        )

        if not clip_paths:
            logger.warning("no_clips_found dir=%s", clips_dir)
            return

        for clip_path in clip_paths:
            camera_id = _parse_camera_id(clip_path.name)
            process_clip(
                clip_path=clip_path,
                store_id=args.store_id,
                camera_id=camera_id,
                layout_path=layout_path,
                emitter=emitter,
                model=model,
                reid_manager=reid_manager,
                billing_queue=billing_queue,
                confidence_threshold=args.conf,
                frame_skip=args.frame_skip,
            )

    logger.info("detection_pipeline_complete")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Store Intelligence Detection Pipeline")
    parser.add_argument("--clips", required=True, help="Directory containing .mp4 clip files")
    parser.add_argument("--layout", required=True, help="Path to store_layout.json")
    parser.add_argument("--store-id", required=True, help="Store ID (e.g. STORE_BLR_002)")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", default=None, help="Optional JSONL file to also write events to")
    parser.add_argument("--model", default="yolov8s.pt", help="YOLOv8 model weights")
    parser.add_argument("--conf", type=float, default=0.40, help="Detection confidence threshold")
    parser.add_argument("--frame-skip", type=int, default=3, help="Process every Nth frame")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
