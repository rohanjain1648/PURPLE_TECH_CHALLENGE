# Architectural Choices

Three decisions that shaped this system, each with options considered, what AI suggested, and what I chose and why.

---

## Decision 1 — Detection Model: YOLOv8s + ByteTrack

### Options Considered

| Option | mAP (person) | Latency @ 1080p | Re-ID built-in | Notes |
|---|---|---|---|---|
| YOLOv8n + ByteTrack | ~37 mAP | ~12ms GPU, ~90ms CPU | No | Too many missed detections in partial occlusion |
| **YOLOv8s + ByteTrack** | ~48 mAP | ~22ms GPU, ~160ms CPU | No | **Chosen** |
| YOLOv8m + ByteTrack | ~52 mAP | ~48ms GPU, ~400ms CPU | No | 4× slower than 's' for +4 mAP |
| RT-DETR-l | ~53 mAP | ~35ms GPU | No | Better detection, no ByteTrack integration |
| YOLOv9 | ~53 mAP | ~40ms GPU | No | Good but no mature tracking integration |

### What AI Suggested

I asked Claude: "For detecting and tracking people in 1080p 15fps retail CCTV footage with a mix of occlusions, which YOLO variant would you recommend?" It suggested YOLOv8m as the default, noting RT-DETR-l as the ceiling if accuracy is paramount. It flagged that ByteTrack outperforms DeepSORT for crowded scenes (billing queue buildup) because ByteTrack's second-pass matching retains low-confidence detections.

### What I Chose and Why

YOLOv8s. The jump from 'n' to 's' is large (+11 mAP); the jump from 's' to 'm' is small (+4 mAP) at 2.5× the compute. With `frame_skip=3` reducing load to 5fps effective, 's' fits on a single 8GB VRAM GPU (or runs on CPU at ~1fps for testing). At 40 live stores I'd consider RT-DETR-l on GPU-equipped edge boxes per store — but for a single-machine evaluation setup, 's' is the pragmatic choice.

**VLM for zone classification:** I evaluated prompting GPT-4V with "Which zone of a retail store is this person standing in?" on frame crops. Accuracy was ~72% (many zones look similar without context). Rule-based polygon intersection against `store_layout.json` gives 100% accuracy as long as the zone polygon calibration is correct. I use the VLM approach only for staff detection (where the uniform is the signal), not zone classification.

**VLM for staff detection:** I tried Claude Vision with the prompt: "Is the person in this image wearing a retail staff uniform (solid-colour, name badge, lanyard)? Answer YES or NO." On 50 manually-labelled frames: 86% precision, 78% recall. The bottleneck is API latency (~1.5s per frame), which is incompatible with 5fps throughput. A locally-quantised LLaVA-1.5 at 200ms/frame is borderline acceptable. I shipped the HSV-histogram approach (< 1ms) as primary, with the contextual heuristic as backup — documented in DESIGN.md. The VLM approach would be my next experiment with a proper GPU.

---

## Decision 2 — Event Schema Design

### Options Considered

**Option A — Flat schema:** All fields at top level. Simple to query, verbose for optional fields.

**Option B — Nested metadata object (chosen):** Variable fields (`queue_depth`, `sku_zone`, `session_seq`) in a `metadata` sub-object. Matches the spec exactly.

**Option C — Event-type-specific schemas:** Separate Pydantic models per event type (EntryEvent, ZoneDwellEvent, etc.). Maximum type safety, but 8 models to maintain and the ingest endpoint would need a discriminated union.

### What AI Suggested

Claude suggested Option C (discriminated union with per-type models) for maximum type safety, noting that it would make downstream analytics queries self-documenting. It also suggested keeping `confidence` optional with a default of 1.0.

### What I Chose and Why

Option B — nested metadata, matching the spec verbatim. Two reasons I overrode the AI:

1. **Spec compliance.** The scoring harness runs against the schema defined in the problem statement. Deviating to Option C, even if architecturally cleaner, risks breaking the automated tests.

2. **Confidence is mandatory.** I disagreed with the AI's suggestion to make `confidence` optional. The spec says "do not suppress low-conf events" — making confidence mandatory forces the pipeline to always report it, preventing accidental silent drops. The model validator enforces `0 ≤ confidence ≤ 1`.

**Re-entry semantics:** The AI initially proposed creating a new `visitor_id` on every re-entry (treating it as a fresh session). I overrode this: the Re-ID system matches the same person to the same `visitor_id`, emits a `REENTRY` event, and opens a new session row with `is_reentry=True`. This is the only way to prevent inflating unique-visitor counts while still tracking re-entry behaviour.

**Schema invariant enforced at ingest:** `ZONE_DWELL` / `ZONE_ENTER` / `ZONE_EXIT` / `BILLING_*` events must have a non-null `zone_id`. The Pydantic `model_validator` rejects events that violate this. Partial success means valid events in the same batch are still stored.

---

## Decision 3 — API Architecture: Sync vs Async, PostgreSQL vs SQLite

### Options Considered

| Option | Pros | Cons |
|---|---|---|
| Flask + SQLAlchemy sync + SQLite | Simplest to run locally | Blocks on every DB query; single-writer SQLite contention under load |
| **FastAPI + SQLAlchemy async + PostgreSQL** | Non-blocking I/O, production-grade concurrency, real DB | Requires Docker Compose to set up |
| FastAPI + async + SQLite (dev only) | No Docker, instant start | WAL mode helps but concurrent writes still block |
| FastAPI + asyncpg direct (no ORM) | Maximum DB performance | Raw SQL maintenance burden; harder to test |

### What AI Suggested

Claude recommended FastAPI with async SQLAlchemy throughout. It specifically suggested using `asyncpg` directly (no ORM) for maximum throughput at the 40-store scale mentioned in the follow-up question examples. It noted that SQLAlchemy ORM adds overhead for bulk-insert operations.

### What I Chose and Why

FastAPI + SQLAlchemy 2.0 async + PostgreSQL (production) / SQLite (dev/test). I partially agreed with the AI:

- **Agreed on FastAPI async.** Correct for an I/O-bound service; lets the event loop handle 40+ concurrent ingest batches without thread pools.
- **Disagreed on dropping the ORM.** At this scale (not millions of rows/second), SQLAlchemy ORM's overhead is acceptable and the benefit — typed models, migrations via Alembic later, test/prod DB interchangeability — outweighs it. The `asyncpg` driver is used under the hood anyway.
- **Added SQLite fallback.** AI didn't suggest this. I added it because the acceptance gate requires `docker compose up` to work, but engineers running tests shouldn't need Docker. `aiosqlite` with the same models makes the test suite run in-process with no external dependencies.

**At 40 stores scale:** The follow-up question asks what breaks first. Answer: the `_count_distinct_visitors` function in `funnel.py` does Python-side de-duplication by loading session rows into memory. At 40 live stores × 1000 visitors/day, that's ~40,000 rows loaded per funnel request. The fix is a proper `COUNT(DISTINCT visitor_id)` subquery in SQL — trivial to add but deliberately deferred to keep the initial implementation readable. Database indices on `(store_id, entry_time)` are already in place.
