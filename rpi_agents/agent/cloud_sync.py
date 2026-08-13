"""Best-effort push of wake-cycle events to the cloud dashboard (F03 design;
ADR-0001, ADR-0006, ADR-0009, ADR-0014, ADR-0015).

Top-level imports remain hardware/network-free; requests and cv2 are
imported lazily, mirroring agent/notifier.py's convention.

Never delays or blocks agent/power.py::resleep() meaningfully (Tenet 1):
push() and flush_queue() catch every exception internally and return,
never raise. A failed push is queued locally (agent/sync_queue.py, bounded,
ADR-0015) and retried on a later wake cycle rather than lost outright.
"""

import base64
import logging
import os
import time

import numpy as np

from agent import config, sync_queue

logger = logging.getLogger(__name__)

_CROCKFORD_ALPHABET = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

_PAYLOAD_FIELDS = (
    "event_id",
    "ts_wall",
    "woken_by_trigger",
    "escalate",
    "motion",
    "person",
    "score",
    "vision_source",
    "is_intrusion",
    "window_broken",
    "alarm",
    "reason",
    "email_sent",
    "latency_s",
)


def generate_event_id() -> str:
    """Generate a ULID: 48-bit ms timestamp + 80-bit randomness, Crockford
    base32 encoded (26 chars) -- sortable, unique, avoids clock-skew
    collisions between the Pi and Azure clocks (F02 design, Data model).

    Identical algorithm to cloud/app/storage.py::generate_ulid() on the
    server side (ADR-0014). The two can't share an import -- different
    processes, different machines -- so keep both copies in sync if either
    ever changes (T04 plan, Notes).
    """
    timestamp_ms = int(time.time() * 1000)
    randomness = int.from_bytes(os.urandom(10), "big")
    value = (timestamp_ms << 80) | randomness
    chars = [""] * 26
    for i in range(25, -1, -1):
        chars[i] = _CROCKFORD_ALPHABET[value & 0x1F]
        value >>= 5
    return "".join(chars)


def _encode_jpeg_b64(snapshot: np.ndarray) -> str:
    """Encode an RGB snapshot to base64 JPEG.

    Mirrors agent/notifier.py::_encode_jpeg's RGB->BGR flip before
    cv2.imencode (that helper is private to notifier.py; this is a
    deliberate, small duplication rather than a cross-module import of a
    "private" function).
    """
    import cv2  # type: ignore[import-untyped]

    ok, buf = cv2.imencode(".jpg", snapshot[..., ::-1])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def build_payload(record: dict, snapshot: np.ndarray | None) -> dict:
    """Build the `POST /api/events` body from machine.py's event record.

    `record` must already include `event_id` (generated once in
    `machine.py`, at decision time -- the same value is reused on every
    retry of this event, immediate or queued; ADR-0014). Mirrors F02's
    Table entity fields minus the server-assigned ones
    (`PartitionKey`/`RowKey`/`received_at`/`blob_name`), plus
    `image_jpeg_b64` only when `snapshot` is not `None` (F01 design;
    ADR-0006).
    """
    payload = {field: record[field] for field in _PAYLOAD_FIELDS}
    if snapshot is not None:
        payload["image_jpeg_b64"] = _encode_jpeg_b64(snapshot)
    return payload


def is_configured() -> bool:
    """True if cloud sync is enabled and has a URL to push to -- i.e.
    push()/flush_queue() will actually attempt work rather than no-op.

    Used by machine.py to decide whether a failed push is even worth
    queuing (ADR-0015): queuing while sync is disabled/unconfigured would
    just accumulate entries that can never succeed.
    """
    return config.CLOUD_SYNC_ENABLED and bool(config.CLOUD_SYNC_URL)


def push(payload: dict) -> bool:
    """Best-effort POST to config.CLOUD_SYNC_URL.

    Returns True on 2xx, False on any failure (disabled, unconfigured,
    missing credential, timeout, connection error, non-2xx). Never raises
    -- unlike agent/notifier.py::notify(), which is allowed to raise and is
    caught by its caller, a failed cloud push is the caller's job to queue
    (agent/machine.py does so explicitly), not push()'s job to retry
    itself.
    """
    event_id = payload.get("event_id")

    if not is_configured():
        return False

    try:
        settings = config.load_settings()
    except ValueError:
        logger.warning(
            "cloud_sync: required secrets missing; skipping push (event_id=%s)", event_id
        )
        return False

    if not settings.cloud_sync_user or not settings.cloud_sync_password:
        logger.warning(
            "cloud_sync: cloud_sync_user/password not configured; skipping push (event_id=%s)",
            event_id,
        )
        return False

    import requests

    try:
        response = requests.post(
            config.CLOUD_SYNC_URL,
            json=payload,
            auth=(settings.cloud_sync_user, settings.cloud_sync_password),
            timeout=config.CLOUD_SYNC_TIMEOUT_S,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("cloud_sync: push failed (event_id=%s): %s", event_id, exc)
        return False

    logger.debug("cloud_sync: pushed event_id=%s", event_id)
    return True


def flush_queue(max_items: int | None = None) -> None:
    """Attempt to push up to `max_items` queued payloads, oldest first.

    Stops at the first failure in this call -- a genuinely dead network
    fails every later attempt identically, so continuing only spends
    timeout budget for nothing (ADR-0015, Risks). Never raises.
    """
    if max_items is None:
        max_items = config.SYNC_QUEUE_MAX_FLUSH_PER_CYCLE

    for entry in sync_queue.dequeue_batch(max_items):
        event_id = entry.get("event_id")
        if push(entry["payload"]):
            sync_queue.remove(event_id)
        else:
            sync_queue.record_failure(event_id)
            break
