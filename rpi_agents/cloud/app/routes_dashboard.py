"""Server-rendered dashboard pages (F04 design).

Reads the same in-process data-access functions F01's API routes use
(`storage.list_events`, `storage.get_event`) — no network hop, no CORS,
since it's the same app/process (ADR-0008). Auth is not referenced here —
`require_basic_auth` is applied once, globally, in `main.py` (F05 design).
"""

from datetime import UTC, datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from . import metrics as metrics_module
from . import storage

router = APIRouter(tags=["dashboard"])

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

_MSG_EVENT_NOT_FOUND = "event not found"


def compute_metrics(events: list[dict]) -> dict:
    """Summary metrics band (F04 design, "Summary metrics band").

    *(ADR-0016)* Delegates to `cloud/app/metrics.py::summary_metrics()` —
    the same computation `GET /api/metrics` now also exposes as JSON, so
    there is one implementation, not two. Kept as a thin wrapper here
    (rather than inlining `metrics_module.summary_metrics` at call sites)
    so `tests/test_dashboard.py`'s existing `from
    cloud.app.routes_dashboard import compute_metrics` import keeps working
    unchanged.
    """
    return metrics_module.summary_metrics(events)


def _with_iso_timestamp(event: dict) -> dict:
    """Add a template-friendly `ts_wall_iso` field — keeps the Jinja
    templates free of custom filters/timezone logic.
    """
    return {
        **event,
        "ts_wall_iso": datetime.fromtimestamp(event["ts_wall"], tz=UTC).isoformat(
            timespec="seconds"
        ),
    }


@router.get("/", response_class=HTMLResponse)
def event_list_page(request: Request) -> HTMLResponse:
    """Event list + metrics band (F04 design, Views 1-2), last 30 days."""
    events = [_with_iso_timestamp(event) for event in storage.list_events()]
    metrics = compute_metrics(events)
    return _templates.TemplateResponse(
        request, "event_list.html", {"events": events, "metrics": metrics}
    )


@router.get("/events/{event_id}", response_class=HTMLResponse)
def event_detail_page(request: Request, event_id: str) -> HTMLResponse:
    """Event detail page (F04 design, View 3)."""
    event = storage.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=_MSG_EVENT_NOT_FOUND)
    return _templates.TemplateResponse(
        request, "event_detail.html", {"event": _with_iso_timestamp(event)}
    )
