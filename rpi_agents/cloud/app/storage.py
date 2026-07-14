"""Data-access layer for Table Storage (`events`) and Blob Storage
(`snapshots`) — F02 design. The only module in `cloud/app` that touches
`azure-data-tables`/`azure-storage-blob` directly (T02 plan, Step 3);
`routes_api.py`/`routes_dashboard.py` call the functions below instead of
holding a Storage credential themselves.
"""

import base64
import binascii
import logging
import os
import time
from datetime import UTC, datetime, timedelta

from azure.data.tables import TableClient, UpdateMode
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas

_TABLE_NAME = "events"
_CONTAINER_NAME = "snapshots"
_SAS_EXPIRY_MINUTES = 15
_DEFAULT_SINCE_DAYS = 30
_DEFAULT_LIMIT = 100
_MAX_LIMIT = 500

_CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_ENV_CONNECTION_STRING = "AZURE_STORAGE_CONNECTION_STRING"

_logger = logging.getLogger(__name__)

_table_client: TableClient | None = None
_blob_service_client: BlobServiceClient | None = None


def _connection_string() -> str:
    conn_str = os.getenv(_ENV_CONNECTION_STRING)
    if not conn_str:
        raise RuntimeError(
            f"{_ENV_CONNECTION_STRING} is not set — see cloud/infra's "
            "storage-connection-string Container App secret (ADR-0010)."
        )
    return conn_str


def get_table_client() -> TableClient:
    """Lazily-constructed, process-wide `TableClient` for the `events` table."""
    global _table_client
    if _table_client is None:
        _table_client = TableClient.from_connection_string(_connection_string(), _TABLE_NAME)
    return _table_client


def get_blob_service_client() -> BlobServiceClient:
    """Lazily-constructed, process-wide `BlobServiceClient`."""
    global _blob_service_client
    if _blob_service_client is None:
        _blob_service_client = BlobServiceClient.from_connection_string(_connection_string())
    return _blob_service_client


def generate_ulid() -> str:
    """Generate a ULID: 48-bit ms timestamp + 80-bit randomness, Crockford
    base32 encoded (26 chars) — sortable, unique, avoids clock-skew
    collisions between the Pi and Azure clocks (F02 design, Data model:
    RowKey). Self-contained; no external ULID dependency.
    """
    timestamp_ms = int(time.time() * 1000)
    randomness = int.from_bytes(os.urandom(10), "big")
    value = (timestamp_ms << 80) | randomness
    chars = [""] * 26
    for i in range(25, -1, -1):
        chars[i] = _CROCKFORD_ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


def _partition_key(ts_wall: float) -> str:
    """UTC calendar date (`YYYY-MM-DD`) of an epoch-seconds timestamp —
    Table entity PartitionKey (F02 design, Data model).
    """
    return datetime.fromtimestamp(ts_wall, tz=UTC).strftime("%Y-%m-%d")


def write_event(fields: dict, event_id: str, received_at: float) -> dict:
    """Write the Table entity for one ingested event; returns the full entity.

    `fields` is `EventIn.model_dump(exclude={"image_jpeg_b64"})` — called
    before `write_blob` (F01 design, Behavior): a metadata-only event is
    still useful even if the image write below fails.
    """
    entity = {
        "PartitionKey": _partition_key(fields["ts_wall"]),
        "RowKey": event_id,
        "received_at": received_at,
        "blob_name": "",
        **{k: v for k, v in fields.items() if v is not None},
    }
    get_table_client().create_entity(entity=entity)
    return entity


def write_blob(event_id: str, image_b64: str) -> str | None:
    """Decode + upload the snapshot image; returns the blob name, or `None`
    on failure. Never raises — a failed blob write must not fail the whole
    ingest request (F01 design, Behavior).
    """
    blob_name = f"{event_id}.jpg"
    try:
        data = base64.b64decode(image_b64, validate=True)
        container = get_blob_service_client().get_container_client(_CONTAINER_NAME)
        container.upload_blob(name=blob_name, data=data, overwrite=True)
    except (binascii.Error, ValueError, OSError):
        _logger.exception("blob write failed for event_id=%s", event_id)
        return None
    return blob_name


def set_blob_name(event_id: str, partition_key: str, blob_name: str) -> None:
    """Patch the Table entity's `blob_name` after a successful blob write."""
    get_table_client().update_entity(
        mode=UpdateMode.MERGE,
        entity={"PartitionKey": partition_key, "RowKey": event_id, "blob_name": blob_name},
    )


def mint_sas_url(blob_name: str, expiry_minutes: int = _SAS_EXPIRY_MINUTES) -> str | None:
    """Mint a fresh, short-lived read-only SAS URL for a snapshot blob.

    Never cached/persisted (F01 design, "GET /api/events") — a new token is
    generated on every call. Returns `None` if `blob_name` is empty (no
    snapshot was pushed for this event).
    """
    if not blob_name:
        return None
    client = get_blob_service_client()
    sas_token = generate_blob_sas(
        account_name=client.account_name,
        container_name=_CONTAINER_NAME,
        blob_name=blob_name,
        account_key=client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.now(UTC) + timedelta(minutes=expiry_minutes),
    )
    return f"{client.url.rstrip('/')}/{_CONTAINER_NAME}/{blob_name}?{sas_token}"


def list_events(since: str | None = None, limit: int = _DEFAULT_LIMIT) -> list[dict]:
    """List events, most recent first, since a given UTC date (inclusive).

    `since` defaults to 30 days ago (F01 design, "GET /api/events"). Table
    Storage has no server-side sort/complex query support (F02 design,
    Risks) — this queries by PartitionKey range (cheap: ISO dates sort
    lexicographically) and does the recency sort + limit client-side, which
    is fine at this project's hobby-scale volume.
    """
    limit = min(max(limit, 1), _MAX_LIMIT)
    since_date = since or (datetime.now(UTC) - timedelta(days=_DEFAULT_SINCE_DAYS)).strftime(
        "%Y-%m-%d"
    )
    query_filter = f"PartitionKey ge '{_escape_odata_literal(since_date)}'"
    entities = get_table_client().query_entities(query_filter=query_filter)
    events = sorted(entities, key=lambda e: e["ts_wall"], reverse=True)
    return [_to_summary_dict(e) for e in events[:limit]]


def get_event(event_id: str) -> dict | None:
    """Fetch one event by its `event_id` (Table RowKey).

    No PartitionKey is available from the URL alone, so this queries by
    RowKey (a filtered scan, not a point-read) — acceptable at this
    project's volume; revisit if the table ever grows past hobby scale.
    """
    query_filter = f"RowKey eq '{_escape_odata_literal(event_id)}'"
    matches = list(get_table_client().query_entities(query_filter=query_filter))
    if not matches:
        return None
    return _to_summary_dict(matches[0])


def _escape_odata_literal(value: str) -> str:
    """Escape a string for safe interpolation into an OData `query_filter`
    (single quotes double, the OData string-literal escape convention) —
    defense-in-depth against filter injection via `since`/`event_id`, both
    of which reach here as caller-controlled strings.
    """
    return value.replace("'", "''")


def _to_summary_dict(entity: dict) -> dict:
    """Table entity -> the dict shape `schemas.EventSummary` expects, with a
    freshly minted SAS `image_url` in place of the raw `blob_name`.
    """
    return {
        "event_id": entity["RowKey"],
        "ts_wall": entity["ts_wall"],
        "woken_by_trigger": entity.get("woken_by_trigger", False),
        "escalate": entity.get("escalate", False),
        "motion": entity.get("motion", False),
        "person": entity.get("person", False),
        "score": entity.get("score", 0.0),
        "vision_source": entity.get("vision_source"),
        "is_intrusion": entity.get("is_intrusion"),
        "alarm": entity.get("alarm", False),
        "reason": entity.get("reason", ""),
        "email_sent": entity.get("email_sent", False),
        "latency_s": entity.get("latency_s", 0.0),
        "received_at": entity.get("received_at", 0.0),
        "image_url": mint_sas_url(entity.get("blob_name", "")),
    }
