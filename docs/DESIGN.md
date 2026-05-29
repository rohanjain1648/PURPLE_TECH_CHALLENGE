# Store Intelligence System — Architecture Design

## System Overview

The system converts raw CCTV footage into live business analytics via a four-stage pipeline:

```
Raw CCTV Clips
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  Detection Layer  (pipeline/)                                   │
│  YOLOv8s → ByteTrack → Zone Mapper → Staff Detector → Re-ID    │
│  Output: structured StoreEvent objects (JSONL + API calls)      │
└────────────────────────────┬────────────────────────────────────┘
                             │  POST /events/ingest (batches ≤500)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Intelligence API  (app/)                                       │
│  FastAPI + SQLAlchemy async → PostgreSQL                        │
│  Endpoints: /metrics  /funnel  /heatmap  /anomalies  /health    │
│  Background: anomaly detection loop every 30 s                  │
└────────────────────────────┬────────────────────────────────────┘
                             │  GET /stores/{id}/metrics (2s poll)
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  Live Dashboard  (dashboard/)                                   │
│  Rich terminal with real-time metrics, heatmap, funnel, alerts  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Detection Layer

### Model: YOLOv8s + ByteTrack

**Why YOLOv8s:** The 's' variant (640px input, ~48 mAP COCO) hits the right balance: fast enough for 15fps on a single mid-range GPU, accurate enough for person detection in typical retail lighting. YOLOv8n would drop too many partial occlusions; YOLOv8m adds ~40% compute for marginal gain on pedestrian detection specifically.

**Why ByteTrack:** ByteTrack (built into ultralytics) uses a two-stage matching strategy — it considers low-confidence detections in the second pass, which is critical for the partial-occlusion cases in billing queues. DeepSORT requires a separate re-ID model at every frame, which doubles latency. StrongSORT gives better occlusion handling but needs more VRAM.

**Frame skip:** Every 3rd frame (15fps → 5fps effective). Reduces CPU 3×. People in retail walk slowly enough that 5fps captures all zone transitions with ≤0.6 s delay. Overridable with `--frame-skip 1` for GPU setups.

### Zone Mapping

Shapely `Polygon.contains(Point(cx, cy))` tested against zone polygons from `store_layout.json`. O(n_zones) per tracked person per frame — negligible for ≤10 zones per camera. Entry/exit uses a virtual crossing line: sign of the 2D cross-product changes when the centroid moves from one side to the other.

### Staff Detection

**Primary (colour-based):** Extracts the dominant HSV colour from the lower 2/3 of each bounding box (clothing region). Compares against staff uniform HSV ranges in `store_layout.json`. Takes ~0.5ms per crop on CPU.

**Contextual (fallback):** If a track traverses >60% of distinct zones within 3 minutes, it is classified as staff. This catches staff visible in off-uniform situations (e.g., manager in plain clothes, delivery person).

**VLM considered:** I tested prompting Claude Vision with "Is this person wearing a retail uniform? Answer YES/NO." Accuracy was ~85% on the sample frames, but latency was 800ms–2s per crop via API — unacceptable at 5fps. A locally-quantised vision model (LLaVA-1.5 7B) gave similar accuracy at 200ms/crop, which is borderline. Documented in CHOICES.md.

### Re-ID and Re-entry Detection

Appearance descriptor: 32-bin LAB colour histogram per channel = 96-dim unit-norm vector. Cosine similarity ≥0.82 → same person. Fast (< 1ms per comparison) and lighting-semi-invariant (LAB separates luminance from chrominance). Gallery of exited descriptors is TTL-pruned at 5 minutes.

**Re-entry edge case:** When similarity is high and the time gap from last exit is < 5 minutes, a REENTRY event is emitted (not a second ENTRY). The Re-ID gallery ensures the same `visitor_id` is reused, so the conversion funnel doesn't double-count.

### Cross-camera Deduplication

Persons visible in both entry and floor cameras (overlap zone) are matched by appearance similarity + temporal proximity (< 30s). The floor camera's track adopts the visitor_id assigned by the entry camera.

### Group Entry

ByteTrack naturally assigns separate track IDs to each person in a group (they have distinct bounding boxes). The zone mapper and Re-ID manager each see individual tracks. The pipeline emits one ENTRY event per track, correctly counting 3 people as 3 visitors.

---

## Intelligence API

### Stack

- **FastAPI** with async handlers — handles concurrent requests without blocking on DB I/O
- **SQLAlchemy 2.0 async** with `asyncpg` (PostgreSQL) or `aiosqlite` (SQLite for dev/test)
- **Pydantic v2** for schema validation — `model_validator` enforces zone_id required for zone-events
- **structlog** for JSON-structured logging with `trace_id` context var per request

### Idempotency

`POST /events/ingest` bulk-fetches existing `event_id`s for the batch in one query, then skips duplicates. A second identical POST returns `accepted=0, duplicates=N` — safe to retry.

### Session State Machine

Events are applied to a sessions table row on ingest:

```
ENTRY  → create/find session row
ZONE_ENTER/EXIT/DWELL → update zones_visited, total_dwell_ms
BILLING_QUEUE_JOIN → set billing_entry_time, queue_joined=True
BILLING_QUEUE_ABANDON → set queue_abandoned=True
EXIT → set exit_time
REENTRY → is_reentry=True on session
```

### POS Correlation

When a POS transaction is ingested, we scan all open sessions for the same store where `billing_entry_time` falls in the 5-minute window before the transaction timestamp. Matched sessions are marked `converted=True`. This is the foundation of the conversion rate metric.

### Anomaly Detection Background Task

An `asyncio.Task` runs every 30 seconds and upserts anomaly records. The `/anomalies` endpoint reads resolved/unresolved records — anomalies persist until conditions clear. Auto-resolution: BILLING_QUEUE_SPIKE resolves when queue drops below threshold. DEAD_ZONE and CONVERSION_DROP re-evaluate each cycle.

### Graceful Degradation

All route handlers wrap DB calls in try/except. If the DB is unavailable, the endpoint returns HTTP 503 with a structured `{"code": "DB_UNAVAILABLE", "message": "..."}` body — no stack trace leaked.

---

## AI-Assisted Decisions

### 1. Event schema design — AI suggested, partially overridden

I asked Claude to draft the event schema for zone-level retail analytics. It proposed a flatter schema without the nested `metadata` object. I pushed back: the spec explicitly requires `metadata.queue_depth` and `metadata.session_seq`, and keeping variable fields in a typed nested object (rather than top-level optionals) makes schema evolution safer. The final schema follows the spec's nested structure.

**Where I agreed:** AI suggested making `confidence` mandatory (not optional) so low-confidence detections surface rather than being silently dropped. I agreed and added the `confidence` field with a `[0, 1]` constraint — it's now required in schema validation.

### 2. Anomaly detection implementation — AI outlined, I restructured

The initial AI-generated anomaly detector ran all checks synchronously in the request path. I moved it to a background task (every 30s) to avoid adding latency to the `/anomalies` GET. The AI draft also wrote DEAD_ZONE as a real-time query per request, which would be expensive at scale. Precomputing into a table that the endpoint reads from is the correct production pattern.

### 3. Storage engine selection — AI recommended PostgreSQL, I added SQLite fallback

AI recommended PostgreSQL throughout. I agreed for production but added an `aiosqlite` fallback (same SQLAlchemy models, different driver) so the test suite runs without Docker and so an engineer can develop locally in under 2 minutes without setting up a database. The `DATABASE_URL` env var controls which backend is used.

---

## Production Readiness Notes

| Concern | Implementation |
|---|---|
| Containerisation | `docker compose up` starts DB + API + pipeline + dashboard |
| Structured logging | structlog + JSON format, `trace_id` bound per request |
| Idempotency | Bulk `event_id` dedup on every ingest batch |
| 503 on DB failure | Try/except in every route handler, structured error body |
| Test coverage | >70% with pytest-cov; edge cases: empty store, all-staff, zero purchases, re-entry |
| Health endpoint | Per-store last-event timestamp + STALE_FEED warning |
