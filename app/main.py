"""
FastAPI application entrypoint.
Wires together all route handlers, middleware, startup/shutdown lifecycle,
and the anomaly-detection background task.
"""
from __future__ import annotations

import asyncio
import time
import uuid
import structlog
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession

from app.anomalies import anomaly_detection_loop, get_active_anomalies
from app.config import get_settings
from app.database import close_db, get_db, init_db
from app.funnel import get_funnel
from app.health import get_health
from app.ingestion import ingest_events, ingest_pos_transactions
from app.logging_config import configure_logging
from app.metrics import get_heatmap, get_store_metrics
from app.models import (
    AnomaliesResponse,
    ErrorDetail,
    EventIn,
    FunnelResponse,
    HealthResponse,
    HeatmapResponse,
    IngestResult,
    PosLoadRequest,
    StoreMetrics,
)

settings = get_settings()
configure_logging(debug=settings.debug)
logger = structlog.get_logger(__name__)

_anomaly_task: asyncio.Task | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator:
    global _anomaly_task
    await init_db()
    _anomaly_task = asyncio.create_task(anomaly_detection_loop())
    logger.info("api_started", version=settings.api_version)
    yield
    if _anomaly_task:
        _anomaly_task.cancel()
        try:
            await _anomaly_task
        except asyncio.CancelledError:
            pass
    await close_db()
    logger.info("api_stopped")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Middleware: structured request logging with trace_id
# ---------------------------------------------------------------------------

@app.middleware("http")
async def logging_middleware(request: Request, call_next) -> Response:
    trace_id = str(uuid.uuid4())[:8]
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id=trace_id)

    start = time.monotonic()
    try:
        response = await call_next(request)
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "request",
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            latency_ms=latency_ms,
            store_id=request.path_params.get("store_id"),
        )
        response.headers["X-Trace-ID"] = trace_id
        return response
    except Exception as exc:
        latency_ms = round((time.monotonic() - start) * 1000, 1)
        logger.error("request_error", path=request.url.path, error=str(exc), latency_ms=latency_ms)
        raise


# ---------------------------------------------------------------------------
# Exception handlers — no raw stack traces in responses
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, _exc: Exception) -> JSONResponse:
    logger.exception("unhandled_error", endpoint=request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorDetail(code="INTERNAL_ERROR", message="An unexpected error occurred").model_dump(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post(
    "/events/ingest",
    response_model=IngestResult,
    status_code=status.HTTP_200_OK,
    summary="Ingest a batch of up to 500 events (idempotent by event_id)",
)
async def ingest(request: Request, db: AsyncSession = Depends(get_db)) -> IngestResult:
    """
    Accepts raw JSON so we can do per-event validation and return partial success
    instead of a 422 for the whole batch when one event is malformed.
    Batch size limit of 500 is enforced manually.
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid JSON body")

    raw_events = body.get("events", [])
    if not isinstance(raw_events, list) or len(raw_events) == 0:
        raise HTTPException(status_code=422, detail="'events' must be a non-empty list")
    if len(raw_events) > settings.max_events_per_batch:
        raise HTTPException(status_code=422, detail=f"Batch exceeds maximum of {settings.max_events_per_batch} events")

    valid: list[EventIn] = []
    pre_errors: list[dict] = []
    for raw in raw_events:
        try:
            valid.append(EventIn.model_validate(raw))
        except (ValidationError, Exception) as exc:
            pre_errors.append({"event_id": raw.get("event_id", "?"), "error": str(exc)})

    structlog.contextvars.bind_contextvars(event_count=len(valid))
    try:
        result = await ingest_events(valid, db)
    except Exception as exc:
        logger.error("ingest_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(code="DB_UNAVAILABLE", message="Storage layer unavailable").model_dump(),
        )
    # Merge schema-level rejections with ingestion-level rejections
    result.rejected += len(pre_errors)
    result.errors = pre_errors + result.errors
    return result


@app.get(
    "/stores/{store_id}/metrics",
    response_model=StoreMetrics,
    summary="Real-time store metrics for today",
)
async def store_metrics(store_id: str, db: AsyncSession = Depends(get_db)) -> StoreMetrics:
    structlog.contextvars.bind_contextvars(store_id=store_id)
    try:
        return await get_store_metrics(store_id, db)
    except Exception as exc:
        logger.error("metrics_failed", store_id=store_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorDetail(code="DB_UNAVAILABLE", message="Storage layer unavailable").model_dump(),
        )


@app.get(
    "/stores/{store_id}/funnel",
    response_model=FunnelResponse,
    summary="Conversion funnel: Entry → Zone → Billing → Purchase",
)
async def store_funnel(store_id: str, db: AsyncSession = Depends(get_db)) -> FunnelResponse:
    structlog.contextvars.bind_contextvars(store_id=store_id)
    try:
        return await get_funnel(store_id, db)
    except Exception as exc:
        logger.error("funnel_failed", store_id=store_id, error=str(exc))
        raise HTTPException(status_code=503, detail=ErrorDetail(code="DB_UNAVAILABLE", message="Storage unavailable").model_dump())


@app.get(
    "/stores/{store_id}/heatmap",
    response_model=HeatmapResponse,
    summary="Zone visit frequency and dwell heatmap (0-100 normalized)",
)
async def store_heatmap(store_id: str, db: AsyncSession = Depends(get_db)) -> HeatmapResponse:
    structlog.contextvars.bind_contextvars(store_id=store_id)
    try:
        return await get_heatmap(store_id, db)
    except Exception as exc:
        logger.error("heatmap_failed", store_id=store_id, error=str(exc))
        raise HTTPException(status_code=503, detail=ErrorDetail(code="DB_UNAVAILABLE", message="Storage unavailable").model_dump())


@app.get(
    "/stores/{store_id}/anomalies",
    response_model=AnomaliesResponse,
    summary="Active operational anomalies with severity and suggested action",
)
async def store_anomalies(store_id: str, db: AsyncSession = Depends(get_db)) -> AnomaliesResponse:
    structlog.contextvars.bind_contextvars(store_id=store_id)
    try:
        return await get_active_anomalies(store_id, db)
    except Exception as exc:
        logger.error("anomalies_failed", store_id=store_id, error=str(exc))
        raise HTTPException(status_code=503, detail=ErrorDetail(code="DB_UNAVAILABLE", message="Storage unavailable").model_dump())


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service health — STALE_FEED if any store has >10 min lag",
)
async def health(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    return await get_health(db)


@app.post(
    "/pos/ingest",
    summary="Load POS transactions for conversion correlation",
)
async def pos_ingest(request: PosLoadRequest, db: AsyncSession = Depends(get_db)) -> dict:
    try:
        result = await ingest_pos_transactions(request.transactions, db)
        return result
    except Exception as exc:
        logger.error("pos_ingest_failed", error=str(exc))
        raise HTTPException(status_code=503, detail=ErrorDetail(code="DB_UNAVAILABLE", message="Storage unavailable").model_dump())


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {"service": settings.api_title, "version": settings.api_version, "docs": "/docs"}
