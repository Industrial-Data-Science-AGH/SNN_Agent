"""Serial events -> contract SpikeBatch payloads. Standard library only; no I/O, no clock.

The Uno reports one decision per 10 ms hop, so the source timeline is a hop grid anchored once to the
Uno's unwrapped micros(): hop k starts at origin + round(k * hop / fs). Batches are whole numbers of hops
laid on that grid, so they tile the timeline exactly; a hole between two batches is a real, explicit gap
(the backend reports it as `missing_batch`). Nothing is interpolated or padded.

Conventions that downstream code must know:
- A spike is placed at the START of the hop it summarises, quantised to the encoder hop (the simulation
  timeline puts frame k at k * dt). Its `dt_us` is the offset from the batch start.
- Priming hops and hops with no spike become observed silence, exactly as in the simulation.
- A merged (late) frame carries one decision for several hops: the spikes go on its last hop and every
  earlier merged hop is counted in `dropped_events`, so the batch is reported as degraded.
- `adc_clipped` is always False: the firmware does not report clipping.
A serial reset is a new boot and therefore a new session; use a new assembler for it.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Sequence

from rpi_agents.agent.serial_protocol import BootEvent, Event, FrameEvent, GapEvent, RejectedEvent

MAX_BATCH_US = 1_000_000
SCHEMA_VERSION = "1.0"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")


@dataclass(frozen=True)
class BatchDraft:
    """A finished batch that does not yet know its session."""

    batch_seq: int
    boot_id: str
    source_start_us: int
    source_end_us: int
    spikes: tuple[tuple[int, str], ...]  # (dt_us, channel), time ordered
    dropped_events: int


@dataclass
class AssemblerStats:
    frames: int = 0
    priming_frames: int = 0
    merged_frames: int = 0
    gap_hops: int = 0
    dropped_hops: int = 0
    spikes: int = 0
    batches: int = 0


class BatchAssembler:
    def __init__(self, *, boot_id: str, channels: Sequence[str], fs_hz: int, hop: int, batch_us: int = 250_000):
        if not _ID.fullmatch(boot_id) or not channels or fs_hz <= 0 or hop <= 0:
            raise ValueError("invalid boot_id, channels, fs_hz or hop")
        self._boot_id, self._channels, self._fs, self._hop = boot_id, tuple(channels), fs_hz, hop
        self._hops_per_batch = max(1, (batch_us * fs_hz + hop * 500_000) // (hop * 1_000_000))
        if self._hops_per_batch * hop * 1_000_000 > MAX_BATCH_US * fs_hz:
            raise ValueError(f"batch_us={batch_us} exceeds the {MAX_BATCH_US} us contract limit")
        self.stats = AssemblerStats()
        self._origin: int | None = None
        self._cursor = 0  # next hop index expected
        self._start: int | None = None  # first hop of the open batch
        self._spikes: list[tuple[int, int]] = []  # (hop, channel index)
        self._dropped = 0
        self._next_seq = 0
        self._first_hop = 0

    @property
    def stream_start_us(self) -> int | None:
        """Start of the first observed hop: the `source_start_us` for creating the session."""
        return None if self._origin is None else self._at(self._first_hop)

    def _grid(self, hop_index: int) -> int:
        return (hop_index * self._hop * 1_000_000 + self._fs // 2) // self._fs

    def _at(self, hop_index: int) -> int:
        return self._origin + self._grid(hop_index)

    def feed(self, event: Event) -> list[BatchDraft]:
        if isinstance(event, RejectedEvent):
            return []
        if isinstance(event, BootEvent):
            raise ValueError("boot changed: a new boot needs a new session and a new BatchAssembler")
        if event.boot_id != self._boot_id:
            raise ValueError(f"event from boot {event.boot_id}, assembler is for {self._boot_id}")
        return self._on_gap(event) if isinstance(event, GapEvent) else self._on_frame(event)

    def flush(self) -> list[BatchDraft]:
        """Close the open batch early (shutdown, or the caller's latency limit)."""
        return self._close(self._cursor) if self._origin is not None else []

    def _on_gap(self, gap: GapEvent) -> list[BatchDraft]:
        if self._origin is None:
            return []
        out = self._close(self._cursor)
        self.stats.gap_hops += gap.missing_hops
        self._cursor = gap.first_seq + gap.missing_hops
        return out

    def _on_frame(self, frame: FrameEvent) -> list[BatchDraft]:
        if frame.mask >> len(self._channels):
            raise ValueError(f"mask {frame.mask:#x} has bits beyond {len(self._channels)} channels")
        first = frame.seq - frame.covered_hops + 1
        if self._origin is None:
            self._origin = max(0, frame.source_us - self._grid(frame.seq + 1))
            self._cursor = self._first_hop = first
        if frame.seq < self._cursor:
            return []  # already covered
        out: list[BatchDraft] = []
        if first > self._cursor:  # a hole the tracker did not announce
            out += self._close(self._cursor)
            self.stats.gap_hops += first - self._cursor
        first = max(first, self._cursor)
        self.stats.frames += 1
        self.stats.priming_frames += frame.priming
        self.stats.merged_frames += frame.covered_hops > 1
        for hop_index in range(first, frame.seq + 1):
            if self._start is None:
                self._start = hop_index
            if hop_index < frame.seq:
                self._dropped += 1
                self.stats.dropped_hops += 1
            else:
                self._spikes += [(hop_index, c) for c in range(len(self._channels)) if frame.mask >> c & 1]
            if hop_index + 1 - self._start >= self._hops_per_batch:
                out += self._close(hop_index + 1)
        self._cursor = frame.seq + 1
        return out

    def _close(self, end_hop: int) -> list[BatchDraft]:
        start = self._start
        if start is None or end_hop <= start:
            self._start, self._spikes, self._dropped = None, [], 0
            return []
        begin_us = self._at(start)
        spikes = tuple((self._at(h) - begin_us, self._channels[c]) for h, c in sorted(self._spikes))
        draft = BatchDraft(self._next_seq, self._boot_id, begin_us, self._at(end_hop), spikes, self._dropped)
        self._next_seq += 1
        self.stats.batches += 1
        self.stats.spikes += len(spikes)
        self._start, self._spikes, self._dropped = None, [], 0
        return [draft]


def request_id(session_id: str, batch_seq: int) -> str:
    """Stable per (session, batch) so an exact retry is idempotent; unique across sessions of a device."""
    rid = f"{session_id}.b{batch_seq}"
    if len(rid) > 64:
        rid = f"{hashlib.sha256(session_id.encode()).hexdigest()[:40]}.b{batch_seq}"
    return rid


def to_spike_batch(
    draft: BatchDraft, *, device_id: str, session_id: str, epoch: int, encoder_hash: str
) -> dict:
    """Stamp a draft with its session identity; the result is a v1 SpikeBatch payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id(session_id, draft.batch_seq),
        "device_id": device_id,
        "session_id": session_id,
        "epoch": epoch,
        "boot_id": draft.boot_id,
        "batch_seq": draft.batch_seq,
        "encoder_hash": encoder_hash,
        "source_start_us": draft.source_start_us,
        "source_end_us": draft.source_end_us,
        "spikes": [{"dt_us": dt, "channel": channel} for dt, channel in draft.spikes],
        "quality": {"dropped_events": draft.dropped_events, "adc_clipped": False},
    }
