"""Uno -> Pi serial line protocol v1: parsing, framing and continuity tracking.

Standard library only (same rule as ports.py); no I/O, no clock, no hardware. The wire format is
the one proposed to the firmware owner for K2:

    $B,<ver>,<build_id>,<fs_hz>,<hop>,<n_ch>,<pulse_us>,<chset>*<CRC>
    $F,<seq>,<t_us>,<n>,<mask>,<flags>,<txdrop>*<CRC>

CRC-8 (poly 0x07, init 0, no reflection, no final XOR) covers the bytes between '$' and '*'.
Lines start with '$' and end with LF or CRLF; lines starting with '#' are firmware comments.
Every rejected line becomes an explicit event; a lost frame surfaces as a GapEvent, never silence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterator

PROTOCOL_VERSION = 1
MAX_LINE_BYTES = 48  # longest valid line, terminator excluded
ASSEMBLER_LIMIT = 64  # receive buffer bound; longer lines are dropped up to the next LF
TIME_SKEW_WARN_US = 2000

_U16 = 0xFFFF
_U32 = 0xFFFFFFFF
_NUM = rb"(0|[1-9][0-9]*)"
_HEX = rb"([0-9A-F]{2})"

_BOOT = re.compile(
    rb"\$B," + _NUM + rb",([0-9A-F]{8})," + _NUM + b"," + _NUM + b"," + _NUM + b"," + _NUM
    + rb",([a-z0-9_]{1,12})\*" + _HEX
)
_FRAME = re.compile(
    rb"\$F," + _NUM + b"," + _NUM + b"," + _NUM + rb",([0-9A-F]{2}),([0-9A-F]{1,2})," + _NUM
    + rb"\*" + _HEX
)


class ProtocolError(ValueError):
    """A line that must not be trusted. `code` is stable for logs and counters."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


@dataclass(frozen=True)
class BootLine:
    version: int
    build_id: str
    fs_hz: int
    hop: int
    n_ch: int
    pulse_us: int
    chset: str


@dataclass(frozen=True)
class FrameLine:
    seq: int
    t_us: int
    n: int
    mask: int
    flags: int
    txdrop: int

    @property
    def priming(self) -> bool:
        return bool(self.flags & 1)


def parse_line(raw: bytes) -> BootLine | FrameLine | None:
    """Parse one line (terminator already removed, or a single trailing CR tolerated).

    Returns None for firmware comments and blank lines; raises ProtocolError for anything else that
    is not a valid, CRC-correct, in-range protocol line.
    """
    line = raw[:-1] if raw.endswith(b"\r") else raw
    if not line or line.startswith(b"#"):
        return None
    if len(line) > MAX_LINE_BYTES:
        raise ProtocolError("TOO_LONG", f"line is {len(line)} bytes, limit {MAX_LINE_BYTES}")
    if not line.startswith(b"$"):
        raise ProtocolError("NOT_PROTOCOL", "line does not start with '$' or '#'")
    pattern = _BOOT if line[1:2] == b"B" else _FRAME if line[1:2] == b"F" else None
    match = pattern.fullmatch(line) if pattern else None
    if match is None:
        raise ProtocolError("BAD_FORMAT", "line does not match the protocol grammar")
    body = line[1 : line.rindex(b"*")]
    if crc8(body) != int(match.group(match.lastindex), 16):
        raise ProtocolError("BAD_CRC", "checksum mismatch")
    fields = [g.decode("ascii") for g in match.groups()[:-1]]
    return _boot(fields) if pattern is _BOOT else _frame(fields)


def _boot(f: list[str]) -> BootLine:
    version, build_id, fs_hz, hop, n_ch, pulse_us = int(f[0]), f[1], int(f[2]), int(f[3]), int(f[4]), int(f[5])
    if version != PROTOCOL_VERSION:
        raise ProtocolError("BAD_RANGE", f"unsupported protocol version {version}")
    if not (1000 <= fs_hz <= 100_000 and 1 <= hop <= 4096 and 1 <= n_ch <= 8 and 1 <= pulse_us <= 1_000_000):
        raise ProtocolError("BAD_RANGE", "boot parameters out of range")
    return BootLine(version, build_id, fs_hz, hop, n_ch, pulse_us, f[6])


def _frame(f: list[str]) -> FrameLine:
    seq, t_us, n, txdrop = int(f[0]), int(f[1]), int(f[2]), int(f[5])
    if seq > _U32 or t_us > _U32 or txdrop > _U16 or not 1 <= n <= _U16:
        raise ProtocolError("BAD_RANGE", "frame field out of range")
    return FrameLine(seq, t_us, n, int(f[3], 16), int(f[4], 16), txdrop)


@dataclass(frozen=True)
class LineOverflow:
    """The assembler dropped an over-long line instead of growing its buffer."""


class LineAssembler:
    """Bounded byte-stream -> line splitter. Never buffers more than `limit` bytes."""

    def __init__(self, limit: int = ASSEMBLER_LIMIT):
        self._limit = limit
        self._buf = bytearray()
        self._discarding = False

    def feed(self, chunk: bytes) -> Iterator[bytes | LineOverflow]:
        for byte in chunk:
            if byte == 0x0A:
                if self._discarding:
                    self._discarding = False
                    yield LineOverflow()
                else:
                    yield bytes(self._buf)
                self._buf.clear()
            elif not self._discarding:
                if len(self._buf) >= self._limit:
                    self._buf.clear()
                    self._discarding = True
                else:
                    self._buf.append(byte)


@dataclass(frozen=True)
class BootEvent:
    boot_id: str
    boot: BootLine


@dataclass(frozen=True)
class FrameEvent:
    boot_id: str
    seq: int
    source_us: int  # Uno micros() unwrapped to 64 bit, monotonic within one boot_id
    n: int
    mask: int
    priming: bool
    covered_hops: int
    txdrop: int
    anomalies: tuple[str, ...] = ()


@dataclass(frozen=True)
class GapEvent:
    boot_id: str
    first_seq: int
    missing_hops: int
    txdrop_delta: int
    cause: str  # "tx_drop" when the Uno reported dropping that many lines, else "lost_or_corrupt"


@dataclass(frozen=True)
class RejectedEvent:
    code: str
    detail: str


Event = BootEvent | FrameEvent | GapEvent | RejectedEvent


class StreamTracker:
    """Turns parsed lines into continuity-checked events.

    Every valid '$B' starts a new boot (new boot_id); the caller only asks the Uno to reprint it
    ('I') when it has no boot context, never mid-run. Frames before a boot line, or after a seq
    regression, are rejected until the next '$B'.
    """

    def __init__(self, new_boot_id: Callable[[], str]):
        self._new_boot_id = new_boot_id
        self._boot: BootLine | None = None
        self._boot_id = ""
        self._last: FrameLine | None = None
        self._next_hop = 0
        self._source_us = 0

    @property
    def synced(self) -> bool:
        return self._boot is not None

    def feed_line(self, raw: bytes) -> list[Event]:
        """One line in, zero or more events out (a gap is reported before the frame that reveals it)."""
        try:
            parsed = parse_line(raw)
        except ProtocolError as exc:
            return [RejectedEvent(exc.code, f"{exc}: {raw[:40]!r}")]
        if parsed is None:
            return []
        return self._boot_line(parsed) if isinstance(parsed, BootLine) else self._frame_line(parsed)

    def _boot_line(self, boot: BootLine) -> list[Event]:
        self._boot, self._boot_id, self._last = boot, self._new_boot_id(), None
        return [BootEvent(self._boot_id, boot)]

    def _frame_line(self, frame: FrameLine) -> list[Event]:
        boot = self._boot
        if boot is None:
            return [RejectedEvent("NO_BOOT", "frame received before a boot line")]
        if frame.mask >> boot.n_ch:
            return [RejectedEvent("BAD_RANGE", f"mask {frame.mask:#x} exceeds {boot.n_ch} channels")]
        last = self._last
        if last is not None and frame.seq == last.seq:
            return [RejectedEvent("SEQ_DUPLICATE", f"seq {frame.seq} repeated")]
        if last is not None and frame.seq < last.seq:
            self._boot = None
            detail = f"seq {frame.seq} after {last.seq}: reset without boot line"
            return [RejectedEvent("SEQ_REGRESSION", detail)]

        covered = max(1, round(frame.n / boot.hop))
        first_covered = frame.seq - covered + 1
        anomalies: list[str] = []
        gap: GapEvent | None = None
        if last is None:
            self._source_us = frame.t_us
            txdrop_delta = 0
        else:
            txdrop_delta = (frame.txdrop - last.txdrop) & _U16
            missing = first_covered - self._next_hop
            if missing > 0:
                cause = "tx_drop" if txdrop_delta >= missing else "lost_or_corrupt"
                gap = GapEvent(self._boot_id, self._next_hop, missing, txdrop_delta, cause)
            elif missing < 0:
                anomalies.append("N_SEQ_MISMATCH")
            self._source_us += (frame.t_us - last.t_us) & _U32
            hop_us = boot.hop * 1_000_000 / boot.fs_hz
            if abs(((frame.t_us - last.t_us) & _U32) - (frame.seq - last.seq) * hop_us) > TIME_SKEW_WARN_US:
                anomalies.append("TIME_SKEW")
        if covered > 1:
            anomalies.append("MERGED")
        if txdrop_delta:
            anomalies.append("TXDROP")
        self._last, self._next_hop = frame, frame.seq + 1
        event = FrameEvent(
            self._boot_id, frame.seq, self._source_us, frame.n, frame.mask, frame.priming,
            covered, frame.txdrop, tuple(anomalies),
        )
        return [gap, event] if gap else [event]
