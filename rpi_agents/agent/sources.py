"""Byte sources and the serial event stream. Standard library only (os, select, termios).

A source hands raw bytes to the protocol layer; the event stream turns them into continuity-checked
events and adds one thing the protocol cannot: a StallEvent when the port is open but no valid frame
has arrived, so a silent Uno is a visible state and never mistaken for silence in the audio.

Opening a serial port normally pulses DTR and resets an Arduino: every open is a new boot and therefore
a new session. Callers must build a fresh StreamTracker after a reconnect.
"""

from __future__ import annotations

import os
import select
import termios
import time
import tty
from dataclasses import dataclass
from typing import Callable, Iterator, Protocol

from rpi_agents.agent.serial_protocol import (
    BootEvent,
    Event,
    FrameEvent,
    LineAssembler,
    LineOverflow,
    RejectedEvent,
    StreamTracker,
)


class SourceDisconnected(ConnectionError):
    """The device went away or the port failed; reconnect and treat the stream as restarted."""


class SourceEnded(Exception):
    """A finite source (replay) is exhausted. This is a normal end, not a fault."""


class ByteSource(Protocol):
    def read(self, max_bytes: int, timeout_s: float) -> bytes:
        """Return up to max_bytes; b'' means nothing arrived within timeout_s."""
        ...

    def close(self) -> None: ...


class ReplaySource:
    """Serves a fixed byte string in small chunks, ending with SourceEnded."""

    def __init__(self, data: bytes, chunk: int = 64):
        if chunk <= 0:
            raise ValueError("chunk must be positive")
        self._data, self._chunk, self._pos = data, chunk, 0

    def read(self, max_bytes: int, timeout_s: float) -> bytes:
        if self._pos >= len(self._data):
            raise SourceEnded
        end = min(self._pos + min(self._chunk, max_bytes), len(self._data))
        piece, self._pos = self._data[self._pos : end], end
        return piece

    def close(self) -> None:
        self._pos = len(self._data)


class SerialPortSource:
    """Raw 8N1 tty reader. Use a persistent path such as /dev/serial/by-id/..., never /dev/ttyACM0."""

    def __init__(self, path: str, baud: int = 115200):
        speed = getattr(termios, f"B{baud}", None)
        if speed is None:
            raise ValueError(f"unsupported baud rate {baud}")
        try:
            fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        except OSError as exc:
            raise SourceDisconnected(f"cannot open {path}: {exc}") from exc
        try:
            tty.setraw(fd, termios.TCSANOW)
            attrs = termios.tcgetattr(fd)
            attrs[2] |= termios.CLOCAL | termios.CREAD
            attrs[4] = attrs[5] = speed
            termios.tcsetattr(fd, termios.TCSANOW, attrs)
            termios.tcflush(fd, termios.TCIOFLUSH)
        except (termios.error, OSError) as exc:
            os.close(fd)
            raise SourceDisconnected(f"cannot configure {path}: {exc}") from exc
        self._fd = fd

    def read(self, max_bytes: int, timeout_s: float) -> bytes:
        try:
            ready, _, _ = select.select([self._fd], [], [], timeout_s)
            if not ready:
                return b""
            data = os.read(self._fd, max_bytes)
        except (OSError, ValueError) as exc:
            raise SourceDisconnected(f"serial read failed: {exc}") from exc
        if not data:
            raise SourceDisconnected("serial port closed")
        return data

    def write(self, data: bytes) -> None:
        """Send command bytes to the device (for example 'I' to reprint the boot line)."""
        try:
            os.write(self._fd, data)
        except OSError as exc:
            raise SourceDisconnected(f"serial write failed: {exc}") from exc

    def close(self) -> None:
        try:
            os.close(self._fd)
        except OSError:
            pass


@dataclass(frozen=True)
class StallEvent:
    """The port is open but no valid frame arrived for `idle_s` seconds."""

    idle_s: float


@dataclass(frozen=True)
class TickEvent:
    """An idle poll: nothing arrived, but the caller gets control (to check a stop request, say)."""


StreamEvent = Event | StallEvent | TickEvent


def event_stream(
    source: ByteSource,
    tracker: StreamTracker,
    *,
    poll_s: float = 0.25,
    stall_s: float = 1.0,
    ticks: bool = False,
    clock: Callable[[], float] = time.monotonic,
) -> Iterator[StreamEvent]:
    """Yield events until the source ends (SourceEnded) or fails (SourceDisconnected propagates).

    With `ticks=True` every idle poll also yields a TickEvent."""
    lines = LineAssembler()
    last_progress, stalled = clock(), False
    while True:
        try:
            chunk = source.read(4096, poll_s)
        except SourceEnded:
            return
        for item in lines.feed(chunk):
            if isinstance(item, LineOverflow):
                events: list[Event] = [RejectedEvent("TOO_LONG", "line dropped: exceeded the receive buffer")]
            else:
                events = tracker.feed_line(item)
            for event in events:
                if isinstance(event, (FrameEvent, BootEvent)):
                    last_progress, stalled = clock(), False
                yield event
        idle = clock() - last_progress
        if idle >= stall_s and not stalled:
            stalled = True
            yield StallEvent(idle)
        if ticks and not chunk:
            yield TickEvent()
