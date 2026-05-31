"""
Main detection + tracking script — driven by clips_config.json.

Usage (real clips):
    python -m pipeline.detect \\
        --clips-config data/clips_config.json \\
        --clips-dir "CCTV Footage" \\
        --layout data/store_layout.json \\
        --api-url http://localhost:8000 \\
        --output events.jsonl

Usage (legacy single-directory scan):
    python -m pipeline.detect \\
        --clips path/to/clips/ \\
        --store-id STORE_PRP_001 \\
        --layout data/store_layout.json \\
        --api-url http://localhost:8000

Model: YOLOv8s (CPU-capable, good occlusion handling with ByteTrack).
Frame skip: configurable per clip (default 6 = 30fps → 5fps effective).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2

from pipeline.emit import EventEmitter
from pipeline.reid import ReIDManager
from pipeline.staff_detector import StaffDetector
from pipeline.tracker import StoreTracker
from pipeline.zone_mapper import ZoneMapper

logger = logging.getLogger(__name__)

# Fallback regex for filenames that embed camera ID
_CAM_RE = re.compile(r"(CAM_[A-Z0-9_]+)", re.IGNORECASE)


def _parse_start_time(ts_str: str) -> datetime:
    """Parse ISO-8601 timestamp (with or without tz offset) → UTC-aware datetime."""
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        logger.warning("Cannot parse clip_start_time '%s', defaulting to now", ts_str)
        return datetime.now(tz=timezone.utc)


def _load_model(model_name: str = "yolov8s.pt"):
    try:
        from ultralytics import YOLO
        model = YOLO(model_name)
        logger.info("yolo_model_loaded model=%s", model_name)
        return model
    except ImportError:
        logger.error("ultralytics not installed — pip install -r requirements-pipeline.txt")
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
    clip_start_time: Optional[datetime] = None,
    confidence_threshold: float = 0.35,
    frame_skip: int = 6,
    force_is_staff: bool = False,
) -> dict:
    """Process one clip. Returns summary stats."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        logger.error("cannot_open_clip path=%s", clip_path)
        return {"frames": 0, "processed": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_time = clip_start_time or datetime.now(tz=timezone.utc)

    logger.info(
        "processing_clip clip=%s store=%s camera=%s fps=%.1f frames=%d start=%s",
        clip_path.name, store_id, camera_id, fps, total_frames,
        start_time.isoformat(),
    )

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
        clip_start_time=start_time,
        current_billing_queue=billing_queue,
        force_is_staff=force_is_staff,
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
                classes=[0],             # person class only
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
    logger.info("clip_done camera=%s total=%d processed=%d", camera_id, frame_idx, processed)
    return {"frames": frame_idx, "processed": processed}


def run_from_config(
    clips_config_path: str,
    clips_dir_override: Optional[str],
    layout_path: str,
    api_url: str,
    output_file: Optional[str],
    model_name: str,
    confidence: float,
) -> None:
    """Process clips defined in clips_config.json."""
    with open(clips_config_path, encoding="utf-8") as f:
        config = json.load(f)

    with EventEmitter(api_url=api_url, output_file=output_file, batch_size=50) as emitter:
        model = _load_model(model_name)

        for store_cfg in config.get("stores", []):
            store_id = store_cfg["store_id"]
            clips_dir = Path(clips_dir_override or store_cfg.get("clips_dir", "CCTV Footage"))

            # Shared Re-ID + queue across cameras of the same store session
            reid_manager = ReIDManager(reentry_window_s=300)
            billing_queue: list[int] = []

            # Process clips in order specified (entry first, then floor, then billing)
            for clip_cfg in store_cfg.get("clips", []):
                filename = clip_cfg["filename"]
                clip_path = clips_dir / filename

                if not clip_path.exists():
                    logger.warning("clip_not_found path=%s (skipping)", clip_path)
                    continue

                camera_id = clip_cfg["camera_id"]
                start_ts = _parse_start_time(clip_cfg.get("clip_start_time", ""))
                frame_skip = clip_cfg.get("frame_skip", 6)
                force_staff = clip_cfg.get("force_is_staff", False)

                process_clip(
                    clip_path=clip_path,
                    store_id=store_id,
                    camera_id=camera_id,
                    layout_path=layout_path,
                    emitter=emitter,
                    model=model,
                    reid_manager=reid_manager,
                    billing_queue=billing_queue,
                    clip_start_time=start_ts,
                    confidence_threshold=confidence,
                    frame_skip=frame_skip,
                    force_is_staff=force_staff,
                )

    logger.info("detection_pipeline_complete")


def run_from_dir(
    clips_dir: str,
    store_id: str,
    layout_path: str,
    api_url: str,
    output_file: Optional[str],
    model_name: str,
    confidence: float,
    frame_skip: int,
) -> None:
    """Legacy: scan a directory and infer camera from filename."""
    clips_path = Path(clips_dir)
    priority = {"ENTRY": 0, "FLOOR": 1, "BILLING": 2}
    clip_files = sorted(
        list(clips_path.glob("*.mp4")) + list(clips_path.glob("*.avi")),
        key=lambda p: priority.get(next((k for k in priority if k in p.name.upper()), ""), 3),
    )
    if not clip_files:
        logger.warning("no_clips_found dir=%s", clips_dir)
        return

    with EventEmitter(api_url=api_url, output_file=output_file, batch_size=50) as emitter:
        model = _load_model(model_name)
        reid_manager = ReIDManager(reentry_window_s=300)
        billing_queue: list[int] = []

        for clip_path in clip_files:
            # Try to infer camera_id from filename, fall back to stem
            m = _CAM_RE.search(clip_path.name)
            camera_id = m.group(1).upper() if m else clip_path.stem.upper().replace(" ", "_")

            process_clip(
                clip_path=clip_path,
                store_id=store_id,
                camera_id=camera_id,
                layout_path=layout_path,
                emitter=emitter,
                model=model,
                reid_manager=reid_manager,
                billing_queue=billing_queue,
                confidence_threshold=confidence,
                frame_skip=frame_skip,
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Store Intelligence Detection Pipeline")

    # Config-driven mode (preferred)
    parser.add_argument("--clips-config", help="Path to clips_config.json")
    parser.add_argument("--clips-dir", help="Override clips directory from config")

    # Legacy directory-scan mode
    parser.add_argument("--clips", help="Directory of .mp4 clips (legacy)")
    parser.add_argument("--store-id", help="Store ID for legacy mode")

    # Common args
    parser.add_argument("--layout", default="data/store_layout.json")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--output", default=None, help="Also write events to this JSONL file")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--frame-skip", type=int, default=6)

    args = parser.parse_args()

    if args.clips_config:
        run_from_config(
            clips_config_path=args.clips_config,
            clips_dir_override=args.clips_dir,
            layout_path=args.layout,
            api_url=args.api_url,
            output_file=args.output,
            model_name=args.model,
            confidence=args.conf,
        )
    elif args.clips and args.store_id:
        run_from_dir(
            clips_dir=args.clips,
            store_id=args.store_id,
            layout_path=args.layout,
            api_url=args.api_url,
            output_file=args.output,
            model_name=args.model,
            confidence=args.conf,
            frame_skip=args.frame_skip,
        )
    else:
        parser.error("Provide --clips-config OR both --clips and --store-id")


if __name__ == "__main__":
    main()
