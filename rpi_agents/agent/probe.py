"""Read a serial source for a few seconds and summarise what the bridge would see.

    python -m rpi_agents.agent.probe --port /dev/serial/by-id/usb-... --seconds 8 --send G

A hardware and replay smoke test, not a service: it opens the port (which normally resets an Arduino),
optionally sends command characters, and prints one JSON object. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from collections import Counter
from typing import Callable, Sequence

from rpi_agents.agent.batching import BatchAssembler
from rpi_agents.agent.serial_protocol import BootEvent, FrameEvent, GapEvent, RejectedEvent, StreamTracker
from rpi_agents.agent.sources import (
    ByteSource,
    SerialPortSource,
    SourceEnded,
    StallEvent,
    event_stream,
)

DEFAULT_CHANNELS = ("peak", "peak_cnt", "cv", "zcr", "flux", "hf_lo", "hf_hi")  # W0 demo channel_map order


class _Deadline:
    """Wraps a source: ends the stream after `seconds` and sends scheduled command bytes once each."""

    def __init__(self, source, seconds: float, clock: Callable[[], float], sends: Sequence[tuple[float, bytes]]):
        self._source, self._clock, self._start = source, clock, clock()
        self._end, self._sends = self._start + seconds, sorted(sends)

    def read(self, max_bytes: int, timeout_s: float) -> bytes:
        now = self._clock()
        if now >= self._end:
            raise SourceEnded
        while self._sends and self._sends[0][0] <= now - self._start:
            self._source.write(self._sends.pop(0)[1])
        return self._source.read(max_bytes, min(timeout_s, max(0.0, self._end - now)))

    def close(self) -> None:
        self._source.close()


def probe(
    source: ByteSource,
    seconds: float,
    *,
    sends: Sequence[tuple[float, bytes]] = (),
    clock: Callable[[], float] = time.monotonic,
) -> dict:
    """`sends` is (seconds after start, bytes): a source given commands must have a write() method."""
    started = clock()
    tracker = StreamTracker(lambda: uuid.uuid4().hex[:16])
    assembler: BatchAssembler | None = None
    summary: dict = {"boots": 0, "frames": 0, "priming": 0, "spike_frames": 0, "batches": 0, "spikes": 0}
    gaps, rejected, anomalies = [], Counter(), Counter()
    first_us = last_us = first_seq = last_seq = None
    for event in event_stream(_Deadline(source, seconds, clock, sends), tracker, poll_s=0.1, stall_s=1.0, clock=clock):
        if isinstance(event, BootEvent):
            summary["boots"] += 1
            b = event.boot
            summary["boot"] = {"build_id": b.build_id, "fs_hz": b.fs_hz, "hop": b.hop, "n_ch": b.n_ch, "chset": b.chset}
            channels = DEFAULT_CHANNELS[: b.n_ch]
            assembler = BatchAssembler(boot_id=event.boot_id, channels=channels, fs_hz=b.fs_hz, hop=b.hop)
        elif isinstance(event, FrameEvent):
            summary["frames"] += 1
            summary["priming"] += event.priming
            summary["spike_frames"] += bool(event.mask)
            anomalies.update(event.anomalies)
            first_us, first_seq = (event.source_us, event.seq) if first_us is None else (first_us, first_seq)
            last_us, last_seq = event.source_us, event.seq
        elif isinstance(event, GapEvent):
            gaps.append({"first_seq": event.first_seq, "missing": event.missing_hops, "cause": event.cause})
        elif isinstance(event, RejectedEvent):
            rejected[event.code] += 1
        elif isinstance(event, StallEvent):
            summary["stalls"] = summary.get("stalls", 0) + 1
        if assembler is not None and not isinstance(event, (BootEvent, StallEvent)):
            for draft in assembler.feed(event):
                summary["batches"] += 1
                summary["spikes"] += len(draft.spikes)
    elapsed = clock() - started
    summary |= {"gaps": gaps, "rejected": dict(rejected), "anomalies": dict(anomalies), "seconds": round(elapsed, 2)}
    if first_us is not None and last_seq > first_seq:
        summary["device_hop_ms"] = round((last_us - first_us) / (last_seq - first_seq) / 1000, 3)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", required=True, help="prefer /dev/serial/by-id/...")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--send", default="", help="command characters, e.g. G, T, L or I")
    parser.add_argument("--send-at", type=float, default=3.0, help="seconds after opening (Mega bootloader ~1-2 s)")
    args = parser.parse_args()
    source = SerialPortSource(args.port, args.baud)
    try:
        sends = [(args.send_at + i * 0.2, ch.encode("ascii")) for i, ch in enumerate(args.send)]
        print(json.dumps(probe(source, args.seconds, sends=sends)))
    finally:
        source.close()


if __name__ == "__main__":
    main()
