"""Shared wiring for the backend services. No I/O of its own."""

from __future__ import annotations

import threading
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from rpi_agents.cloud.app.settings import Settings
from rpi_agents.cloud.app.storage import Storage
from rpi_agents.runtime.ports import SNNRuntime


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Context:
    storage: Storage
    settings: Settings
    manifest: dict  # the validated ModelManifest that live sessions must match
    runtime_factory: Callable[[], SNNRuntime]
    clock: Callable[[], datetime] = utcnow
    new_id: Callable[[], str] = lambda: str(uuid.uuid4())
    runtimes: dict[str, SNNRuntime] = field(default_factory=dict)  # session_id -> live runtime (this process)
    _locks: dict[str, threading.RLock] = field(default_factory=lambda: defaultdict(threading.RLock))
    _locks_guard: threading.Lock = field(default_factory=threading.Lock)

    def now(self) -> datetime:
        return self.clock()

    def lock(self, key: str) -> threading.RLock:
        """One lock per key (a device or a session): the single writer that epochs and leases rely on."""
        with self._locks_guard:
            return self._locks[key]
