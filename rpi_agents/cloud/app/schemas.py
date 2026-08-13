"""Pydantic models for the ingest/read API (F01 design, "Contracts").

`EventIn` is a closed schema (`extra="forbid"`) — an unexpected field on an
ingest POST should fail loudly with a 422, not get silently dropped
(F01 design, Validation; ADR-0006). `EventSummary`/`EventDetail` mirror the
Table Storage entity shape from F02's design, with `blob_name` replaced by a
freshly-minted SAS `image_url` — the browser and the Pi's push client never
see a raw blob name or a Storage credential.
"""

import base64
import binascii
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

_MAX_IMAGE_BYTES = 2 * 1024 * 1024
"""2 MB decoded cap on image_jpeg_b64 (F01 design, Validation)."""

_MSG_IMAGE_TOO_LARGE = "image_jpeg_b64 exceeds the 2 MB decoded size cap"
_MSG_IMAGE_NOT_BASE64 = "image_jpeg_b64 is not valid base64"
_MSG_EVENT_ID_INVALID = "event_id must be a 26-character Crockford-base32 ULID (uppercase)"

_ULID_LENGTH = 26
_CROCKFORD_ALPHABET = frozenset("0123456789ABCDEFGHJKMNPQRSTVWXYZ")

VisionSource = Literal["gemini", "failsafe"]


class EventIn(BaseModel):
    """`POST /api/events` request body (F01 design, "POST /api/events").

    `event_id` is client-supplied (ADR-0014): the Pi generates it
    (`agent/cloud_sync.py::generate_event_id()`) so a queued retry can be
    posted twice without creating a duplicate row — the server upserts on
    it instead of minting its own (`storage.write_event()`). `received_at`
    remains server-generated (`utc epoch at ingest`), never trusted from the
    request.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    ts_wall: float
    woken_by_trigger: bool
    escalate: bool
    motion: bool
    person: bool
    score: float
    vision_source: VisionSource | None = None
    is_intrusion: bool | None = None
    window_broken: bool | None = None
    alarm: bool
    reason: str
    email_sent: bool
    latency_s: float
    image_jpeg_b64: str | None = None

    @field_validator("event_id")
    @classmethod
    def _check_event_id(cls, value: str) -> str:
        if len(value) != _ULID_LENGTH or not set(value) <= _CROCKFORD_ALPHABET:
            raise ValueError(_MSG_EVENT_ID_INVALID)
        return value

    @field_validator("image_jpeg_b64")
    @classmethod
    def _check_image_size(cls, value: str | None) -> str | None:
        if value is None:
            return value
        try:
            decoded = base64.b64decode(value, validate=True)
        except binascii.Error as exc:
            raise ValueError(_MSG_IMAGE_NOT_BASE64) from exc
        if len(decoded) > _MAX_IMAGE_BYTES:
            raise ValueError(_MSG_IMAGE_TOO_LARGE)
        return value


class EventIngestResponse(BaseModel):
    """`202` response body for a successful ingest."""

    event_id: str


class EventSummary(BaseModel):
    """One row of `GET /api/events` (F01 design, "GET /api/events").

    Same shape as the Table entity (F02 design, Data model) with
    `PartitionKey`/`RowKey`/`blob_name` replaced by `event_id` and a
    freshly-minted `image_url` — nothing downstream of this app ever holds a
    raw blob name.
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str
    ts_wall: float
    woken_by_trigger: bool
    escalate: bool
    motion: bool
    person: bool
    score: float
    vision_source: VisionSource | None
    is_intrusion: bool | None
    window_broken: bool | None
    alarm: bool
    reason: str
    email_sent: bool
    latency_s: float
    received_at: float
    image_url: str | None
    window_broken_confirmed: bool | None
    intrusion_confirmed: bool | None
    reviewed_at: float | None


class EventDetail(EventSummary):
    """`GET /api/events/{event_id}` response.

    Identical field set to `EventSummary` today — the list route already
    returns every Table field via the summary. Kept as its own type (rather
    than reusing `EventSummary` directly) because F04's detail view is
    expected to be the place debug-only fields land later, and call sites
    should say which contract they mean.
    """


class EventReview(BaseModel):
    """`PATCH /api/events/{event_id}` request body (F01 design, "PATCH
    /api/events/{event_id}"; ADR-0018) — the owner's manual ground-truth
    review of a previously ingested event. Both fields are required: a
    review confirms both judgments together, matching how the owner
    actually reviews an event (look at the photo, decide both "was the
    window really broken" and "was a person actually there" in one pass),
    not a partial update. `reviewed_at` is server-assigned (`utc epoch at
    review time`), same pattern as `received_at` on ingest — never trusted
    from the request.
    """

    model_config = ConfigDict(extra="forbid")

    window_broken_confirmed: bool
    intrusion_confirmed: bool


class SummaryMetrics(BaseModel):
    """`summary` block of `GET /api/metrics` — unchanged from the original
    HTML-only metrics band (F04 design)."""

    model_config = ConfigDict(extra="forbid")

    real_wakes: int
    false_wakes: int
    non_escalating_wakes: int
    email_delivery_rate: float
    gemini_call_success_rate: float
    window_break_confirmation_rate: float


class DailyMetrics(BaseModel):
    """One entry of `GET /api/metrics`'s `daily` list — per-UTC-date
    counts, oldest first (F01 design, "GET /api/metrics")."""

    model_config = ConfigDict(extra="forbid")

    date: str
    real_wakes: int
    false_wakes: int
    non_escalating_wakes: int
    total: int


class VisionSourceCounts(BaseModel):
    """One row of `vision_source_breakdown` — counts for one outcome
    bucket, by which vision path (if any) produced it."""

    model_config = ConfigDict(extra="forbid")

    gemini: int
    failsafe: int
    none: int


class VisionSourceBreakdown(BaseModel):
    """Outcome x `vision_source` cross-tab (ADR-0016) — an operational/
    model-agreement breakdown, **not** a ground-truth confusion matrix (see
    F04 design, Risks: this system never captures whether an event was
    actually an intrusion, only what the pipeline itself decided).
    """

    model_config = ConfigDict(extra="forbid")

    real_wakes: VisionSourceCounts
    false_wakes: VisionSourceCounts
    non_escalating_wakes: VisionSourceCounts


class LatencyStats(BaseModel):
    """avg/p50/p95/max of `latency_s` across the `GET /api/metrics` window."""

    model_config = ConfigDict(extra="forbid")

    avg: float
    p50: float
    p95: float
    max: float


class TriggerOutcomeCounts(BaseModel):
    """One row of `trigger_breakdown` — outcome counts for either genuinely
    SNN-triggered wakes or any other-reason wake."""

    model_config = ConfigDict(extra="forbid")

    real_wakes: int
    false_wakes: int
    non_escalating_wakes: int


class TriggerBreakdown(BaseModel):
    """Outcome x `woken_by_trigger` cross-tab (2026-07-15, ADR-0016
    addendum). The deployed system is designed so every wake is an SNN
    hardware trigger (`agent/trigger.py`) — `not_triggered` should normally
    be all zeros; a nonzero count is an anomaly signal (an unexpected
    manual/dev boot reaching the cloud), not a second normal wake path.
    """

    model_config = ConfigDict(extra="forbid")

    triggered: TriggerOutcomeCounts
    not_triggered: TriggerOutcomeCounts


class LastSync(BaseModel):
    """Most recent event in the queried window, by `received_at`
    (2026-07-15 addition) — lets the dashboard show "last synced X ago"."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    ts_wall: float
    received_at: float


class ConfusionCounts(BaseModel):
    """TP/FP/TN/FN + accuracy of one Gemini-predicted field against the
    owner's manually confirmed ground truth (2026-07-15, ADR-0018)."""

    model_config = ConfigDict(extra="forbid")

    tp: int
    fp: int
    tn: int
    fn: int
    accuracy: float


class ReviewAccuracy(BaseModel):
    """Real confusion-matrix accuracy over manually reviewed events
    (2026-07-15, ADR-0018) — unlike `vision_source_breakdown`/
    `trigger_breakdown`, this genuinely measures correctness against
    owner-confirmed ground truth, not just an operational breakdown. Only
    counts events with a real Gemini verdict AND a completed review;
    `reviewed_count` is 0 (and both confusion blocks all-zero) until the
    owner has reviewed at least one event.
    """

    model_config = ConfigDict(extra="forbid")

    reviewed_count: int
    window_broken: ConfusionCounts
    intrusion: ConfusionCounts


class MetricsResponse(BaseModel):
    """`GET /api/metrics` response (F01 design, "GET /api/metrics";
    ADR-0016)."""

    model_config = ConfigDict(extra="forbid")

    since: str
    until: str
    summary: SummaryMetrics
    daily: list[DailyMetrics]
    vision_source_breakdown: VisionSourceBreakdown
    trigger_breakdown: TriggerBreakdown
    last_sync: LastSync | None
    review_accuracy: ReviewAccuracy
    latency_s: LatencyStats
