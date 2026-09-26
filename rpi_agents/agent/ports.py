"""Small device-side seams; standard library only, no torch/cloud SDK imports.

W1 supplies hardware adapters; W0 never instantiates or energizes hardware.
Device implementations must refuse demo commands and check session/epoch/TTL.
"""

from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(frozen=True)
class CameraImage:
    jpeg: bytes
    captured_at: str  # UTC RFC3339 Z; capture, not upload timestamp


class CameraAdapter(Protocol):
    def capture(self, *, max_bytes: int) -> CameraImage:
        """Capture one bounded JPEG; raise on disconnection or oversize."""
        ...


class SerialAdapter(Protocol):
    def batches(self) -> Iterable[dict]:
        """Yield SpikeBatch-compatible payloads; gaps/restarts must be explicit."""
        ...

    def close(self) -> None: ...


class AlarmAdapter(Protocol):
    def apply(self, *, duration_ms: int, led: bool, buzzer: bool) -> None:
        """Locally enforce maximum duration and automatic off, even without cloud."""
        ...

    def off(self) -> None: ...
