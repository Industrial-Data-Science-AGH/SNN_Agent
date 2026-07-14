"""Bounded local backlog for events that failed to push to the cloud
dashboard (F03 design; ADR-0015).

One JSON object per line in config.SYNC_QUEUE_PATH:
    {"event_id": str, "payload": dict, "attempts": int, "queued_at": float}

`enqueue()` is a true append (mirrors agent/machine.py::_log_event's
EVENT_LOG pattern) unless the queue is already full, in which case making
room requires rewriting the file. `remove()`/`record_failure()` always
rewrite (there's no way to delete/edit one JSONL line in place). Every
rewrite is atomic (write to a temp file, then os.replace) so a crash
mid-write can't corrupt the file into something unreadable; a torn last
line from a true-append interrupted by power loss is simply skipped on
read, never treated as fatal.

This file is a cloud-delivery worklist only -- agent/machine.py's
EVENT_LOG remains the complete, uncapped local record regardless of what
this queue drops.
"""

import json
import logging
import time

from agent import config

logger = logging.getLogger(__name__)


def _read_all() -> list[dict]:
    """Read every well-formed line in the queue file, oldest first.
    Missing file -> []. A line that fails to parse is skipped and logged,
    not treated as fatal (e.g. a write torn by power loss)."""
    if not config.SYNC_QUEUE_PATH.exists():
        return []
    entries = []
    with config.SYNC_QUEUE_PATH.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("sync_queue: skipping malformed line")
    return entries


def _write_all(entries: list[dict]) -> None:
    """Atomically overwrite the queue file with exactly these entries."""
    config.SYNC_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = config.SYNC_QUEUE_PATH.with_suffix(".jsonl.tmp")
    with tmp_path.open("w") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")
    tmp_path.replace(config.SYNC_QUEUE_PATH)


def _append_one(entry: dict) -> None:
    """True single-line append -- no read of the existing file needed."""
    config.SYNC_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.SYNC_QUEUE_PATH.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")


def enqueue(event_id: str, payload: dict) -> None:
    """Append one entry for a push that failed. Drops the oldest entry
    first if already at config.SYNC_QUEUE_MAX_SIZE, to make room
    (ADR-0015) -- that path requires a full rewrite; the common case
    (queue not yet full) is a true append.
    """
    new_entry = {
        "event_id": event_id,
        "payload": payload,
        "attempts": 0,
        "queued_at": time.time(),
    }
    entries = _read_all()
    if len(entries) >= config.SYNC_QUEUE_MAX_SIZE:
        dropped = entries.pop(0)
        logger.warning(
            "sync_queue: full (%d entries); dropping oldest event_id=%s",
            config.SYNC_QUEUE_MAX_SIZE,
            dropped.get("event_id"),
        )
        entries.append(new_entry)
        _write_all(entries)
    else:
        _append_one(new_entry)


def dequeue_batch(max_items: int) -> list[dict]:
    """Return up to max_items oldest queued entries, without removing them.
    Caller removes explicitly via remove() after a confirmed push, or via
    record_failure() once an entry's attempts are exhausted.
    """
    return _read_all()[:max_items]


def remove(event_id: str) -> None:
    """Remove one entry after a confirmed successful push."""
    entries = [entry for entry in _read_all() if entry.get("event_id") != event_id]
    _write_all(entries)


def record_failure(event_id: str) -> None:
    """Increment one entry's attempts counter; drop it (logged) once
    config.SYNC_QUEUE_MAX_ATTEMPTS is reached, rather than let a
    permanently-broken entry block everything behind it forever (the queue
    is always processed oldest-first).
    """
    kept = []
    for entry in _read_all():
        if entry.get("event_id") == event_id:
            entry["attempts"] = entry.get("attempts", 0) + 1
            if entry["attempts"] >= config.SYNC_QUEUE_MAX_ATTEMPTS:
                logger.error(
                    "sync_queue: dropping event_id=%s after %d failed attempts",
                    event_id,
                    entry["attempts"],
                )
                continue
        kept.append(entry)
    _write_all(kept)
