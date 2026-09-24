"""Small durable outbox for edge -> cloud messages. Standard library only (sqlite3).

Guarantees:
- Survives process crashes and restarts: pending entries are re-read in their original order.
- Idempotent by request_id: the same payload again is a no-op, different content raises OutboxConflict
  (the API answers 409 for the same situation).
- Bounded: when `max_pending` is reached the OLDEST pending entry is dropped, counted persistently and
  returned to the caller, so loss becomes an explicit sequence gap on the backend, never silence. The
  newest data is kept because it is the most relevant one for an alarm.
- Delivery order equals insertion order; a consumer must send entries oldest first.
WAL with synchronous=NORMAL survives a process crash without corruption; a sudden power cut can lose the
last few transactions, which the backend then sees as a gap like any other loss.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass

MAX_PAYLOAD_BYTES = 65536  # same limit as contracts.validation.MAX_BATCH_BYTES

_SCHEMA = """
CREATE TABLE IF NOT EXISTS outbox (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT NOT NULL UNIQUE,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    sent INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS outbox_pending ON outbox (sent, id);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
"""


class OutboxConflict(ValueError):
    """The request_id already exists with different content."""


@dataclass(frozen=True)
class Entry:
    id: int
    request_id: str
    kind: str
    payload: dict
    attempts: int
    last_error: str | None


@dataclass(frozen=True)
class PutResult:
    stored: bool  # False for an exact duplicate
    dropped: tuple[str, ...] = ()  # request_ids discarded to stay within max_pending


@dataclass(frozen=True)
class OutboxStats:
    pending: int
    sent: int
    dropped_total: int


def _canonical(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Outbox:
    def __init__(self, path: str, *, max_pending: int = 5000):
        if max_pending < 1:
            raise ValueError("max_pending must be at least 1")
        self._max_pending = max_pending
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def put(self, request_id: str, payload: dict, kind: str = "batch") -> PutResult:
        body = _canonical(payload)
        if len(body.encode()) > MAX_PAYLOAD_BYTES:
            raise ValueError(f"payload is {len(body.encode())} bytes, limit {MAX_PAYLOAD_BYTES}")
        with self._lock:
            row = self._db.execute("SELECT payload FROM outbox WHERE request_id = ?", (request_id,)).fetchone()
            if row is not None:
                if row[0] != body:
                    raise OutboxConflict(f"request_id {request_id} already stored with different content")
                return PutResult(stored=False)
            self._db.execute("BEGIN IMMEDIATE")
            try:
                dropped = self._drop_oldest_over_limit()
                self._db.execute(
                    "INSERT INTO outbox (request_id, kind, payload) VALUES (?, ?, ?)", (request_id, kind, body)
                )
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
            return PutResult(stored=True, dropped=dropped)

    def _drop_oldest_over_limit(self) -> tuple[str, ...]:
        pending = self._db.execute("SELECT COUNT(*) FROM outbox WHERE sent = 0").fetchone()[0]
        excess = pending + 1 - self._max_pending
        if excess <= 0:
            return ()
        rows = self._db.execute(
            "SELECT id, request_id FROM outbox WHERE sent = 0 ORDER BY id LIMIT ?", (excess,)
        ).fetchall()
        self._db.executemany("DELETE FROM outbox WHERE id = ?", [(r[0],) for r in rows])
        self._db.execute(
            "INSERT INTO meta (key, value) VALUES ('dropped_total', ?) "
            "ON CONFLICT(key) DO UPDATE SET value = value + excluded.value",
            (len(rows),),
        )
        return tuple(r[1] for r in rows)

    def pending(self, limit: int = 50) -> list[Entry]:
        """Oldest first. Send strictly in this order."""
        with self._lock:
            rows = self._db.execute(
                "SELECT id, request_id, kind, payload, attempts, last_error FROM outbox "
                "WHERE sent = 0 ORDER BY id LIMIT ?",
                (limit,),
            ).fetchall()
        return [Entry(r[0], r[1], r[2], json.loads(r[3]), r[4], r[5]) for r in rows]

    def record_failure(self, request_id: str, error: str) -> None:
        with self._lock:
            self._db.execute(
                "UPDATE outbox SET attempts = attempts + 1, last_error = ? WHERE request_id = ? AND sent = 0",
                (error[:200], request_id),
            )

    def mark_sent(self, request_id: str) -> bool:
        """Called after the backend acknowledged the entry. False if it was unknown or already sent."""
        with self._lock:
            cursor = self._db.execute(
                "UPDATE outbox SET sent = 1 WHERE request_id = ? AND sent = 0", (request_id,)
            )
        return cursor.rowcount == 1

    def purge_sent(self, keep: int = 100) -> int:
        """Forget delivered entries but keep the newest `keep`, so late duplicates are still recognised."""
        with self._lock:
            cursor = self._db.execute(
                "DELETE FROM outbox WHERE sent = 1 AND id NOT IN "
                "(SELECT id FROM outbox WHERE sent = 1 ORDER BY id DESC LIMIT ?)",
                (keep,),
            )
        return cursor.rowcount

    def stats(self) -> OutboxStats:
        with self._lock:
            pending = self._db.execute("SELECT COUNT(*) FROM outbox WHERE sent = 0").fetchone()[0]
            sent = self._db.execute("SELECT COUNT(*) FROM outbox WHERE sent = 1").fetchone()[0]
            row = self._db.execute("SELECT value FROM meta WHERE key = 'dropped_total'").fetchone()
        return OutboxStats(pending, sent, row[0] if row else 0)
