"""`POST/GET /api/events`, `GET /api/events/{event_id}`, `GET /api/metrics`
(F01 design).

Auth is not referenced anywhere in this module — `require_basic_auth` is
applied once, globally, to the whole app in `main.py` (F05 design: "one
dependency ... applied globally", "not per-route").
"""

import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, Query, status

from . import metrics, schemas, storage

router = APIRouter(prefix="/api/events", tags=["events"])
metrics_router = APIRouter(prefix="/api", tags=["metrics"])

_MSG_EVENT_NOT_FOUND = "event not found"
_ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
_DEFAULT_SINCE_DAYS = 30


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=schemas.EventIngestResponse)
def ingest_event(payload: schemas.EventIn) -> schemas.EventIngestResponse:
    """Ingest one wake-cycle event from the Pi (F01 design, "POST /api/events").

    `event_id` is client-supplied and upserted on (ADR-0014) — a retried
    push of an event the server already committed overwrites in place
    instead of duplicating or erroring. `received_at` is still generated
    here, never trusted from the request. Writes the Table entity first,
    then the blob (if an image was sent) — a blob-write failure leaves
    `blob_name` empty rather than failing the whole request (F01 design,
    Behavior).
    """
    event_id = payload.event_id
    received_at = time.time()
    fields = payload.model_dump(exclude={"image_jpeg_b64", "event_id"})
    entity = storage.write_event(fields, event_id, received_at)

    if payload.image_jpeg_b64 is not None:
        blob_name = storage.write_blob(event_id, payload.image_jpeg_b64)
        if blob_name is not None:
            storage.set_blob_name(event_id, entity["PartitionKey"], blob_name)

    return schemas.EventIngestResponse(event_id=event_id)


@router.get("", response_model=list[schemas.EventSummary])
def list_events(
    since: str | None = Query(
        default=None,
        pattern=_ISO_DATE_PATTERN,
        description="ISO date (YYYY-MM-DD); defaults to 30 days ago",
    ),
    limit: int = Query(default=100, ge=1, le=500),
) -> list[schemas.EventSummary]:
    """List events, most recent first (F01 design, "GET /api/events")."""
    events = storage.list_events(since=since, limit=limit)
    return [schemas.EventSummary(**event) for event in events]


@router.get("/{event_id}", response_model=schemas.EventDetail)
def get_event_detail(event_id: str) -> schemas.EventDetail:
    """Fetch one event's full detail (F01 design, "GET /api/events/{event_id}")."""
    event = storage.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_MSG_EVENT_NOT_FOUND)
    return schemas.EventDetail(**event)


@metrics_router.get("/metrics", response_model=schemas.MetricsResponse)
def get_metrics(
    since: str | None = Query(
        default=None,
        pattern=_ISO_DATE_PATTERN,
        description="ISO date (YYYY-MM-DD); defaults to 30 days ago",
    ),
) -> schemas.MetricsResponse:
    """Dashboard analytics (F01 design, "GET /api/metrics"; ADR-0016).

    Uncapped aggregation over the selected window — unlike `GET
    /api/events`, there is no `limit`: a rollup/breakdown must reflect the
    whole window, not one page of it (accepted scaling trade-off, ADR-0016
    Risks). `vision_source_breakdown` is an operational/model-agreement
    cross-tab, not a ground-truth confusion matrix — this system never
    captures whether an event was actually an intrusion, only what the
    pipeline itself decided (F04 design, Risks).
    """
    since_date = since or (datetime.now(UTC) - timedelta(days=_DEFAULT_SINCE_DAYS)).strftime(
        "%Y-%m-%d"
    )
    until_date = datetime.now(UTC).strftime("%Y-%m-%d")
    events = storage.list_events_for_metrics(since=since_date)
    return schemas.MetricsResponse(
        since=since_date,
        until=until_date,
        summary=metrics.summary_metrics(events),
        daily=metrics.daily_rollup(events),
        vision_source_breakdown=metrics.vision_source_breakdown(events),
        latency_s=metrics.latency_stats(events),
    )
