"""
Event simulator — generates realistic visitor journeys and sends them to the API.
Use this to demo the live dashboard or populate the database for testing
without needing actual CCTV clips.

Usage:
    python -m pipeline.simulate \\
        --store-id STORE_BLR_002 \\
        --layout data/store_layout.json \\
        --api-url http://localhost:8000 \\
        --visitors 50 \\
        --speed 10        # 10x real time

Or for a continuous live feed:
    python -m pipeline.simulate --live --speed 1
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import random
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from pipeline.emit import EventEmitter, make_event

logger = logging.getLogger(__name__)

# Typical zone visit probabilities (tuned to a beauty retail store)
_ZONE_WEIGHTS = {
    "SKINCARE": 0.80,
    "MAKEUP": 0.65,
    "HAIRCARE": 0.45,
    "FRAGRANCE": 0.35,
    "WELLNESS": 0.25,
    "BILLING": 0.40,    # reaches billing if they intend to buy
}

# Avg dwell time per zone in seconds (mean, std)
_ZONE_DWELL = {
    "SKINCARE": (120, 60),
    "MAKEUP": (90, 45),
    "HAIRCARE": (60, 30),
    "FRAGRANCE": (45, 20),
    "WELLNESS": (40, 20),
    "BILLING": (180, 90),
}

_STAFF_RATIO = 0.08          # 8 % of "visitors" are actually staff
_GROUP_ENTRY_PROB = 0.15     # 15 % chance of a group entry (2-4 people)
_REENTRY_PROB = 0.05         # 5 % of visitors re-enter within the session


def _visitor_id(seed: str) -> str:
    h = hashlib.sha256(seed.encode()).hexdigest()[:6]
    return f"VIS_{h}"


def _simulate_visitor_session(
    store_id: str,
    camera_id_entry: str,
    camera_id_floor: str,
    camera_id_billing: str,
    zone_ids: list[str],
    emitter: EventEmitter,
    start_time: datetime,
    is_staff: bool = False,
    visitor_seed: Optional[str] = None,
) -> datetime:
    """Simulate one visitor's journey through the store. Returns the exit time."""
    seed = visitor_seed or str(random.random())
    vid = _visitor_id(seed)
    now = start_time

    # ENTRY
    emitter.emit(make_event(
        store_id=store_id, camera_id=camera_id_entry, visitor_id=vid,
        event_type="ENTRY", timestamp=now, confidence=random.uniform(0.75, 0.99),
        is_staff=is_staff,
    ))

    if is_staff:
        # Staff traverses all zones quickly
        for zone in zone_ids:
            now += timedelta(seconds=random.uniform(10, 30))
            emitter.emit(make_event(
                store_id=store_id, camera_id=camera_id_floor, visitor_id=vid,
                event_type="ZONE_ENTER", timestamp=now, zone_id=zone,
                is_staff=True, confidence=random.uniform(0.80, 0.98),
            ))
            now += timedelta(seconds=random.uniform(5, 15))
            emitter.emit(make_event(
                store_id=store_id, camera_id=camera_id_floor, visitor_id=vid,
                event_type="ZONE_EXIT", timestamp=now, zone_id=zone,
                is_staff=True, confidence=random.uniform(0.80, 0.98),
            ))
    else:
        # Customer visits a random subset of zones
        visited_zones = [z for z in zone_ids if z != "BILLING"
                         and random.random() < _ZONE_WEIGHTS.get(z, 0.5)]
        random.shuffle(visited_zones)

        for zone in visited_zones:
            now += timedelta(seconds=random.uniform(5, 20))  # walking time
            emitter.emit(make_event(
                store_id=store_id, camera_id=camera_id_floor, visitor_id=vid,
                event_type="ZONE_ENTER", timestamp=now, zone_id=zone,
                confidence=random.uniform(0.70, 0.97),
            ))
            dwell_mean, dwell_std = _ZONE_DWELL.get(zone, (60, 30))
            dwell_s = max(10, random.gauss(dwell_mean, dwell_std))

            # Emit ZONE_DWELL events every 30 s during long dwells
            elapsed = 0.0
            while elapsed + 30 < dwell_s:
                now += timedelta(seconds=30)
                elapsed += 30
                emitter.emit(make_event(
                    store_id=store_id, camera_id=camera_id_floor, visitor_id=vid,
                    event_type="ZONE_DWELL", timestamp=now, zone_id=zone,
                    dwell_ms=30000, confidence=random.uniform(0.70, 0.97),
                ))
            now += timedelta(seconds=dwell_s - elapsed)
            emitter.emit(make_event(
                store_id=store_id, camera_id=camera_id_floor, visitor_id=vid,
                event_type="ZONE_EXIT", timestamp=now, zone_id=zone,
                confidence=random.uniform(0.70, 0.97),
            ))

        # Billing decision
        will_buy = random.random() < _ZONE_WEIGHTS["BILLING"]
        if will_buy or visited_zones:
            now += timedelta(seconds=random.uniform(5, 15))
            queue_depth = random.randint(0, 4)
            emitter.emit(make_event(
                store_id=store_id, camera_id=camera_id_billing, visitor_id=vid,
                event_type="ZONE_ENTER", timestamp=now, zone_id="BILLING",
                confidence=random.uniform(0.75, 0.98),
            ))
            if queue_depth > 0:
                emitter.emit(make_event(
                    store_id=store_id, camera_id=camera_id_billing, visitor_id=vid,
                    event_type="BILLING_QUEUE_JOIN", timestamp=now, zone_id="BILLING",
                    queue_depth=queue_depth, confidence=random.uniform(0.75, 0.98),
                ))

            billing_wait = random.gauss(120, 60)
            # Simulate abandonment (20 % if queue > 2)
            if queue_depth > 2 and random.random() < 0.20:
                now += timedelta(seconds=random.uniform(30, 90))
                emitter.emit(make_event(
                    store_id=store_id, camera_id=camera_id_billing, visitor_id=vid,
                    event_type="BILLING_QUEUE_ABANDON", timestamp=now, zone_id="BILLING",
                    confidence=random.uniform(0.70, 0.95),
                ))
            else:
                now += timedelta(seconds=max(30, billing_wait))
                emitter.emit(make_event(
                    store_id=store_id, camera_id=camera_id_billing, visitor_id=vid,
                    event_type="ZONE_EXIT", timestamp=now, zone_id="BILLING",
                    confidence=random.uniform(0.75, 0.98),
                ))

    # EXIT
    now += timedelta(seconds=random.uniform(5, 20))
    emitter.emit(make_event(
        store_id=store_id, camera_id=camera_id_entry, visitor_id=vid,
        event_type="EXIT", timestamp=now, confidence=random.uniform(0.75, 0.99),
        is_staff=is_staff,
    ))
    return now


def run_simulation(
    store_id: str,
    layout_path: str,
    api_url: str,
    num_visitors: int,
    speed: float = 1.0,
    live: bool = False,
    seed: int = 42,
) -> None:
    import json
    random.seed(seed)

    with open(layout_path, encoding="utf-8") as f:
        layout = json.load(f)
    stores = layout.get("stores", [layout])
    store_cfg = next((s for s in stores if s["store_id"] == store_id), stores[0])

    # Extract zone IDs (non-billing product zones)
    zone_ids = [z["zone_id"] for z in store_cfg.get("zones", [])]
    cam_entry = next(
        (c["camera_id"] for c in store_cfg.get("cameras", []) if c.get("type") == "entry"),
        "CAM_ENTRY_01",
    )
    cam_floor = next(
        (c["camera_id"] for c in store_cfg.get("cameras", []) if c.get("type") == "floor"),
        "CAM_FLOOR_01",
    )
    cam_billing = next(
        (c["camera_id"] for c in store_cfg.get("cameras", []) if c.get("type") == "billing"),
        "CAM_BILLING_01",
    )

    with EventEmitter(api_url=api_url, batch_size=20) as emitter:
        sim_time = datetime.now(tz=timezone.utc).replace(hour=10, minute=0, second=0, microsecond=0)
        visitor_count = 0

        while live or visitor_count < num_visitors:
            is_staff = random.random() < _STAFF_RATIO

            # Group entry
            group_size = 1
            if not is_staff and random.random() < _GROUP_ENTRY_PROB:
                group_size = random.randint(2, 4)

            for _ in range(group_size):
                _simulate_visitor_session(
                    store_id=store_id,
                    camera_id_entry=cam_entry,
                    camera_id_floor=cam_floor,
                    camera_id_billing=cam_billing,
                    zone_ids=zone_ids,
                    emitter=emitter,
                    start_time=sim_time,
                    is_staff=is_staff,
                )
                visitor_count += 1

            # Next visitor arrives 20-120 s later (sim time)
            inter_arrival = random.expovariate(1 / 60)  # avg 60s between arrivals
            sim_time += timedelta(seconds=inter_arrival)

            if live:
                # Sleep in real time (divided by speed factor)
                time.sleep(inter_arrival / speed)
            else:
                logger.debug("simulated visitor %d/%d", visitor_count, num_visitors)

        emitter.flush()

    # Also emit POS transactions
    _emit_pos_transactions(store_id, api_url, sim_time)
    logger.info("simulation_complete visitors=%d", visitor_count)


def _emit_pos_transactions(store_id: str, api_url: str, up_to: datetime) -> None:
    """Generate synthetic POS transactions and POST to /pos/ingest."""
    txns = []
    current = up_to - timedelta(hours=8)  # go back to store open
    txn_num = 1
    while current < up_to:
        txns.append({
            "transaction_id": f"SIM_TXN_{txn_num:05d}",
            "store_id": store_id,
            "timestamp": current.isoformat(),
            "basket_value_inr": round(random.uniform(200, 3000), 2),
        })
        current += timedelta(seconds=random.expovariate(1 / 300))  # avg 5 min between txns
        txn_num += 1

    try:
        resp = requests.post(f"{api_url}/pos/ingest", json={"transactions": txns}, timeout=15)
        resp.raise_for_status()
        logger.info("pos_transactions_loaded count=%d", len(txns))
    except Exception as exc:
        logger.warning("pos_ingest_failed error=%s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Store Intelligence Event Simulator")
    parser.add_argument("--store-id", default="STORE_BLR_002")
    parser.add_argument("--layout", default="data/store_layout.json")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--visitors", type=int, default=50)
    parser.add_argument("--speed", type=float, default=60.0, help="Simulation speed multiplier")
    parser.add_argument("--live", action="store_true", help="Run continuously (real-time at --speed)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_simulation(
        store_id=args.store_id,
        layout_path=args.layout,
        api_url=args.api_url,
        num_visitors=args.visitors,
        speed=args.speed,
        live=args.live,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
