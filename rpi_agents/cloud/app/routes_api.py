"""`POST/GET /api/events`, `GET /api/events/{event_id}` (F01 design).

Auth is not referenced anywhere in this module — `require_basic_auth` is
applied once, globally, to the whole app in `main.py` (F05 design: "one
dependency ... applied globally", "not per-route").
"""

import time

from fastapi import APIRouter, HTTPException, Query, status

from . import schemas, storage

router = APIRouter(prefix="/api/events", tags=["events"])

_MSG_EVENT_NOT_FOUND = "event not found"
_ISO_DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"


@router.post("", status_code=status.HTTP_202_ACCEPTED, response_model=schemas.EventIngestResponse)
def ingest_event(payload: schemas.EventIn) -> schemas.EventIngestResponse:
    """Ingest one wake-cycle event from the Pi (F01 design, "POST /api/events").

    `event_id`/`received_at` are generated here, never trusted from the
    request. Writes the Table entity first, then the blob (if an image was
    sent) — a blob-write failure leaves `blob_name` empty rather than
    failing the whole request (F01 design, Behavior).
    """
    event_id = storage.generate_ulid()
    received_at = time.time()
    fields = payload.model_dump(exclude={"image_jpeg_b64"})
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
