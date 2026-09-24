"""Storage interfaces for the backend, with in-memory implementations. Standard library only.

The interfaces deliberately mirror what Azure Table, Blob and Queue Storage really guarantee, no more:
- Tables: entities addressed by (partition, row); optimistic concurrency with ETags; a transaction is atomic
  only within ONE partition. There is no cross-table or table+queue transaction, which is why the backend
  uses an outbox (see events.py) instead of pretending otherwise.
- Blobs: create-only writes are the idempotency primitive (`put` raises Conflict when the name exists).
- Queues: at-least-once delivery. A received message becomes invisible for `visibility_s` and reappears if it
  is not deleted; `renew` extends that, `dequeue_count` tells a worker how often it was already tried.

MemoryStorage is what the tests and local development use. storage_azure.py implements the same interfaces
on Azure; tests/w0/test_storage.py holds one contract suite that any implementation must pass.
"""

from __future__ import annotations

import copy
import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol


class StorageError(Exception):
    """Base class for storage failures that the caller can act on."""


class Conflict(StorageError):
    """The entity or blob already exists."""


class NotFound(StorageError):
    """The entity, blob or queue message does not exist."""


class PreconditionFailed(StorageError):
    """The ETag or the queue receipt is stale: someone else changed or took it."""


@dataclass(frozen=True)
class Row:
    pk: str
    rk: str
    data: dict
    etag: str


@dataclass(frozen=True)
class Op:
    """One step of a single-partition transaction."""

    kind: str  # insert | replace | delete
    rk: str
    data: dict | None = None
    etag: str | None = None


@dataclass(frozen=True)
class QueueMessage:
    id: str
    receipt: str
    body: dict
    dequeue_count: int


class Tables(Protocol):
    def get(self, table: str, pk: str, rk: str) -> Row | None: ...
    def insert(self, table: str, pk: str, rk: str, data: dict) -> Row: ...
    def replace(self, table: str, pk: str, rk: str, data: dict, etag: str) -> Row: ...
    def delete(self, table: str, pk: str, rk: str, etag: str | None = None) -> None: ...
    def query(
        self, table: str, pk: str, *, rk_prefix: str | None = None, rk_from: str | None = None,
        limit: int = 100, reverse: bool = False,
    ) -> list[Row]: ...  # fmt: skip
    def transaction(self, table: str, pk: str, ops: list[Op]) -> list[Row]:
        """All operations apply or none do. One partition, at most 100 operations, and a row may appear only once
        (ValueError otherwise): these are Azure's rules and the in-memory implementation enforces them too."""
        ...


class Blobs(Protocol):
    def put(self, container: str, name: str, data: bytes, content_type: str) -> None:
        """Create-only: raises Conflict when the blob exists."""
        ...

    def get(self, container: str, name: str) -> bytes: ...
    def exists(self, container: str, name: str) -> bool: ...
    def delete(self, container: str, name: str) -> None: ...


class Queues(Protocol):
    def send(self, queue: str, body: dict, *, delay_s: float = 0.0) -> str: ...
    def receive(self, queue: str, *, visibility_s: float, max_messages: int = 1) -> list[QueueMessage]: ...
    def renew(self, queue: str, message: QueueMessage, visibility_s: float) -> QueueMessage: ...
    def delete(self, queue: str, message: QueueMessage) -> None: ...
    def depth(self, queue: str) -> int:
        """Approximate number of messages, visible or not."""
        ...


@dataclass
class Storage:
    tables: Tables
    blobs: Blobs
    queues: Queues


class MemoryTables:
    def __init__(self):
        self._lock = threading.RLock()
        self._data: dict[tuple[str, str], dict[str, tuple[dict, str]]] = {}
        self._counter = itertools.count(1)

    def _partition(self, table: str, pk: str) -> dict[str, tuple[dict, str]]:
        return self._data.setdefault((table, pk), {})

    def _etag(self) -> str:
        return f"W/\"{next(self._counter)}\""

    def get(self, table, pk, rk):
        with self._lock:
            found = self._partition(table, pk).get(rk)
            return None if found is None else Row(pk, rk, copy.deepcopy(found[0]), found[1])

    def insert(self, table, pk, rk, data):
        return self.transaction(table, pk, [Op("insert", rk, data)])[0]

    def replace(self, table, pk, rk, data, etag):
        return self.transaction(table, pk, [Op("replace", rk, data, etag)])[0]

    def delete(self, table, pk, rk, etag=None):
        self.transaction(table, pk, [Op("delete", rk, None, etag)])

    def query(self, table, pk, *, rk_prefix=None, rk_from=None, limit=100, reverse=False):
        with self._lock:
            rows = sorted(self._partition(table, pk).items(), reverse=reverse)
            out = []
            for rk, (data, etag) in rows:
                if rk_prefix is not None and not rk.startswith(rk_prefix):
                    continue
                if rk_from is not None and (rk > rk_from if reverse else rk < rk_from):
                    continue
                out.append(Row(pk, rk, copy.deepcopy(data), etag))
                if len(out) >= limit:
                    break
            return out

    def transaction(self, table, pk, ops):
        if not ops:
            return []
        if len({op.rk for op in ops}) != len(ops):
            raise ValueError("a row can appear only once in a transaction")
        if len(ops) > 100:
            raise ValueError("a transaction is limited to 100 operations")
        with self._lock:
            part = self._partition(table, pk)
            staged = dict(part)
            result: list[Row] = []
            for op in ops:
                current = staged.get(op.rk)
                if op.kind == "insert":
                    if current is not None:
                        raise Conflict(f"{table}/{pk}/{op.rk} exists")
                elif current is None:
                    raise NotFound(f"{table}/{pk}/{op.rk}")
                elif op.etag is not None and op.etag != current[1]:
                    raise PreconditionFailed(f"{table}/{pk}/{op.rk} changed")
                elif op.kind == "replace" and op.etag is None:
                    raise PreconditionFailed("replace needs an etag")
                if op.kind == "delete":
                    staged.pop(op.rk, None)
                    continue
                etag = self._etag()
                staged[op.rk] = (copy.deepcopy(op.data), etag)
                result.append(Row(pk, op.rk, copy.deepcopy(op.data), etag))
            self._data[(table, pk)] = staged  # commit: nothing above raised
            return result


class MemoryBlobs:
    def __init__(self):
        self._lock = threading.Lock()
        self._data: dict[tuple[str, str], tuple[bytes, str]] = {}

    def put(self, container, name, data, content_type):
        with self._lock:
            if (container, name) in self._data:
                raise Conflict(f"{container}/{name} exists")
            self._data[(container, name)] = (bytes(data), content_type)

    def get(self, container, name):
        with self._lock:
            if (container, name) not in self._data:
                raise NotFound(f"{container}/{name}")
            return self._data[(container, name)][0]

    def exists(self, container, name):
        with self._lock:
            return (container, name) in self._data

    def delete(self, container, name):
        with self._lock:
            self._data.pop((container, name), None)


@dataclass
class _Msg:
    id: str
    body: dict
    visible_at: float
    receipt: str = ""
    dequeue_count: int = 0


@dataclass
class _Q:
    messages: list[_Msg] = field(default_factory=list)


class MemoryQueues:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._lock = threading.Lock()
        self._clock = clock
        self._queues: dict[str, _Q] = {}
        self._ids = itertools.count(1)
        self._receipts = itertools.count(1)

    def _q(self, queue: str) -> _Q:
        return self._queues.setdefault(queue, _Q())

    def send(self, queue, body, *, delay_s=0.0):
        with self._lock:
            msg = _Msg(f"m{next(self._ids)}", copy.deepcopy(body), self._clock() + delay_s)
            self._q(queue).messages.append(msg)
            return msg.id

    def receive(self, queue, *, visibility_s, max_messages=1):
        with self._lock:
            now, out = self._clock(), []
            for msg in self._q(queue).messages:
                if msg.visible_at <= now and len(out) < max_messages:
                    msg.dequeue_count += 1
                    msg.receipt = f"r{next(self._receipts)}"
                    msg.visible_at = now + visibility_s
                    out.append(QueueMessage(msg.id, msg.receipt, copy.deepcopy(msg.body), msg.dequeue_count))
            return out

    def _find(self, queue: str, message: QueueMessage) -> _Msg:
        for msg in self._q(queue).messages:
            if msg.id == message.id:
                if msg.receipt != message.receipt:
                    raise PreconditionFailed("stale receipt: the message was received again")
                return msg
        raise NotFound(f"message {message.id}")

    def renew(self, queue, message, visibility_s):
        with self._lock:
            msg = self._find(queue, message)
            msg.visible_at = self._clock() + visibility_s
            msg.receipt = f"r{next(self._receipts)}"
            return QueueMessage(msg.id, msg.receipt, copy.deepcopy(msg.body), msg.dequeue_count)

    def delete(self, queue, message):
        with self._lock:
            self._q(queue).messages.remove(self._find(queue, message))

    def depth(self, queue):
        with self._lock:
            return len(self._q(queue).messages)


def memory_storage(clock: Callable[[], float] = time.monotonic) -> Storage:
    return Storage(MemoryTables(), MemoryBlobs(), MemoryQueues(clock))
