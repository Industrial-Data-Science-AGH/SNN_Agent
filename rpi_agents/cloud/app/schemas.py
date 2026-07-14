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
    alarm: bool
    reason: str
    email_sent: bool
    latency_s: float
    received_at: float
    image_url: str | None


class EventDetail(EventSummary):
    """`GET /api/events/{event_id}` response.

    Identical field set to `EventSummary` today — the list route already
    returns every Table field via the summary. Kept as its own type (rather
    than reusing `EventSummary` directly) because F04's detail view is
    expected to be the place debug-only fields land later, and call sites
    should say which contract they mean.
    """
