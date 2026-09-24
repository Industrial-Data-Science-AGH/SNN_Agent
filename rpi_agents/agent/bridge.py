"""The edge bridge service: Uno serial stream -> durable outbox -> backend. Standard library only.

    python -m rpi_agents.agent.bridge --config /etc/snn-edge/edge.toml

Three threads. The main thread owns the serial stream and never waits on the network, because the kernel
buffer holds only ~0.35 s of Uno data. The worker thread creates sessions, delivers the outbox in order
and sends the heartbeat. The command thread polls and executes capture commands: a photo blocks for about a
second, and that must never delay delivery or make an ack outlive its command.

Every Uno boot (power-up, USB reconnect, reset) is a new session and gets its own BootContext: batches
made before its session exists wait in a bounded buffer, and a closed context finalises itself later
(stamps its buffered batches, then stops its session) without ever blocking the reader. A new session is
created only after every earlier session's stop has been delivered, because the backend allows one active
session per device. Nothing is invented: lost data shows up as a sequence/time gap on the backend, and
overflow of any buffer is counted and reported in the heartbeat.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from rpi_agents.agent.api import ApiClient, Backoff, Outcome, Result, UrllibTransport
from rpi_agents.agent.batching import BatchAssembler, BatchDraft, to_spike_batch
from rpi_agents.agent.commands import CommandHandler, ImageSink, SessionRef
from rpi_agents.agent.config import Config, ConfigError, load_config, read_token
from rpi_agents.agent.outbox import Entry, Outbox
from rpi_agents.agent.ports import CameraAdapter
from rpi_agents.agent.serial_protocol import (
    BootEvent,
    BootLine,
    FrameEvent,
    GapEvent,
    RejectedEvent,
    StreamTracker,
)
from rpi_agents.agent.sources import (
    ByteSource,
    ReplaySource,
    SerialPortSource,
    SourceDisconnected,
    StallEvent,
    TickEvent,
    event_stream,
)

log = logging.getLogger("snn_edge.bridge")
SCHEMA_VERSION = "1.0"
MAX_CLOSED_CONTEXTS = 8
SESSION_RETRY_PERMANENT_S = 30.0
DEFAULT_IMAGE_BYTES, DEFAULT_MAX_FRAMES = 1_048_576, 3


def _rid(prefix: str, ident: str) -> str:
    """A request id within the contract's 64-character pattern, stable for the same input."""
    rid = f"{prefix}-{ident}"
    return rid if len(rid) <= 64 else f"{prefix}-{hashlib.sha256(ident.encode()).hexdigest()[:40]}"


def _utc_z(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class _Create:
    request_id: str
    source_start_us: int
    backoff: Backoff = field(default_factory=lambda: Backoff(0.5, 15.0))
    next_try: float = 0.0
    last_error: str | None = None


@dataclass
class BootContext:
    boot_id: str
    boot: BootLine
    assembler: BatchAssembler | None  # None when the boot was refused (see fault)
    fault: str | None = None
    pre_session: deque = field(default_factory=deque)  # drafts waiting for their session
    create: _Create | None = None  # set once the first frame fixes the session's source_start_us
    session: SessionRef | None = None
    closed: bool = False
    cancelled: bool = False  # abandoned before its session existed; a late creation must be stopped at once


@dataclass
class Counters:
    boots: int = 0
    frames: int = 0
    gaps: int = 0
    rejected: int = 0
    stalls: int = 0
    reconnects: int = 0
    dropped_pre_session: int = 0
    status_failures: int = 0


class Bridge:
    def __init__(
        self,
        config: Config,
        *,
        api: ApiClient,
        state: Outbox,
        source_factory: Callable[[], ByteSource],
        camera: CameraAdapter | None = None,
        sink: ImageSink | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        new_boot_id: Callable[[], str] = lambda: uuid.uuid4().hex[:16],
        agent_version: str = "dev",
        worker_tick_s: float = 0.2,
    ):
        self._cfg, self._api, self._state, self._source_factory = config, api, state, source_factory
        self._camera, self._mono, self._utc_now, self._new_boot_id = camera, monotonic, utc_now, new_boot_id
        self._version, self._tick_s = agent_version, worker_tick_s
        self._lock = threading.Lock()
        self._current: BootContext | None = None
        self._closed: list[BootContext] = []
        self._create_queue: deque[BootContext] = deque()
        self._unstopped: dict[str, str | None] = {}  # session_id -> request_id of its stop, once enqueued
        self._counters = Counters()
        self._connected, self._stalled, self._draining = False, False, False
        self._synced = False  # a boot line has been seen on this connection
        self._last_frame: float | None = None
        self._started = self._mono()
        self._last_boot_request = -1e9
        self._retry_at, self._backoff = 0.0, Backoff(0.5, 30.0)
        self._worker_stop, self._commands_stop = threading.Event(), threading.Event()
        self._handler = CommandHandler(
            device_id=config.device.id, state=state, camera=camera, sink=sink,
            session=self._session_ref, enqueue_ack=lambda ack: self._enqueue(ack["request_id"], ack, "ack"),
            utc_now=utc_now, monotonic=monotonic,
        )  # fmt: skip

    # ------------------------------------------------------------------ public

    @property
    def counters(self) -> Counters:
        return self._counters

    def run(self, stop: threading.Event) -> int:
        """Serve until `stop` is set or a finite (replay) source ends. Returns a process exit code."""
        self._recover_startup()
        worker = threading.Thread(target=self._worker_loop, name="snn-edge-worker", daemon=True)
        commands = threading.Thread(target=self._command_loop, name="snn-edge-commands", daemon=True)
        worker.start()
        commands.start()
        try:
            self._main_loop(stop)
        finally:
            self._shutdown(worker, commands)
        return 0

    # ------------------------------------------------------------- main thread

    def _main_loop(self, stop: threading.Event) -> None:
        backoff, attempts = Backoff(0.5, 5.0), 0
        while not stop.is_set():
            try:
                source = self._source_factory()
            except SourceDisconnected as exc:
                self._connected = False
                log.warning("serial unavailable: %s", exc)
                stop.wait(backoff.next_delay())
                continue
            attempts += 1
            self._connected, self._stalled = True, False
            if attempts > 1:
                self._counters.reconnects += 1
            try:
                outcome = self._serve(source, stop)
            except SourceDisconnected as exc:
                self._connected = False
                log.warning("serial disconnected: %s", exc)
                outcome = "disconnected"
            finally:
                source.close()
            if outcome in ("stopped", "ended"):
                return  # _shutdown closes the boot, after the command thread has stopped
            self._close_current()  # every reconnect is a new boot: never pretend continuity
            stop.wait(backoff.next_delay())

    def _serve(self, source: ByteSource, stop: threading.Event) -> str:
        self._synced = False
        tracker = StreamTracker(self._new_boot_id)
        stream = event_stream(
            source, tracker, poll_s=0.2, stall_s=self._cfg.serial.stall_s, ticks=True, clock=self._mono
        )
        for event in stream:
            self._handle(event, source)
            self._maintain()
            if stop.is_set():
                return "stopped"
        return "ended"

    def _handle(self, event, source: ByteSource) -> None:
        if isinstance(event, TickEvent):
            return
        if isinstance(event, BootEvent):
            self._on_boot(event)
        elif isinstance(event, StallEvent):
            self._counters.stalls += 1
            self._stalled = True
            ctx = self._current
            if ctx is not None and ctx.assembler is not None:
                self._route(ctx, ctx.assembler.flush())  # deliver what was observed before the silence
        elif isinstance(event, RejectedEvent):
            if not self._synced:
                # Before the first boot line the port is still delivering the tail of whatever the device was
                # sending when it was opened: expected, not an error.
                log.debug("ignored before first boot line: %s: %s", event.code, event.detail)
            else:
                self._counters.rejected += 1
                if self._counters.rejected <= 5 or self._counters.rejected % 100 == 0:
                    log.warning("rejected serial line #%d: %s: %s", self._counters.rejected, event.code, event.detail)
            if event.code == "NO_BOOT" and self._mono() - self._last_boot_request > 1.0:
                self._last_boot_request = self._mono()
                write = getattr(source, "write", None)
                if write is not None:
                    write(b"I")  # ask the device to reprint its boot line
        elif isinstance(event, (FrameEvent, GapEvent)):
            self._on_stream(event)

    def _on_boot(self, event: BootEvent) -> None:
        self._synced = True
        self._close_current()
        self._counters.boots += 1
        fault = self._check_boot(event.boot)
        assembler = None
        if fault is None:
            assembler = BatchAssembler(
                boot_id=event.boot_id, channels=self._cfg.serial.channels, fs_hz=event.boot.fs_hz,
                hop=event.boot.hop, batch_us=self._cfg.limits.batch_ms * 1000,
            )  # fmt: skip
        else:
            log.error("refusing boot %s: %s", event.boot_id, fault)
        self._current = BootContext(event.boot_id, event.boot, assembler, fault)
        self._stalled = False

    def _check_boot(self, boot: BootLine) -> str | None:
        if boot.n_ch != len(self._cfg.serial.channels):
            return f"channel map mismatch: firmware reports {boot.n_ch} channels, config lists {len(self._cfg.serial.channels)}"
        expected = self._cfg.serial.expected_build_id
        if expected is not None and boot.build_id != expected:
            return f"unexpected firmware build {boot.build_id} (expected {expected})"
        return None

    def _on_stream(self, event) -> None:
        ctx = self._current
        if ctx is None or ctx.assembler is None:
            return
        if isinstance(event, FrameEvent):
            self._counters.frames += 1
            self._last_frame, self._stalled = self._mono(), False
        else:
            self._counters.gaps += 1
        try:
            drafts = ctx.assembler.feed(event)
        except ValueError as exc:  # protocol/config inconsistency: stop trusting this boot, keep running
            ctx.fault, ctx.assembler = f"batching refused the stream: {exc}", None
            log.error("boot %s: %s", ctx.boot_id, ctx.fault)
            return
        if ctx.create is None and ctx.assembler.stream_start_us is not None:
            ctx.create = _Create(_rid("create", ctx.boot_id), ctx.assembler.stream_start_us)
            with self._lock:
                self._create_queue.append(ctx)
        self._route(ctx, drafts)

    def _route(self, ctx: BootContext, drafts: list[BatchDraft]) -> None:
        self._drain(ctx)
        for draft in drafts:
            if ctx.session is not None:
                self._put_batch(ctx, draft)
                continue
            if len(ctx.pre_session) >= self._cfg.limits.pre_session_batches:
                ctx.pre_session.popleft()
                self._counters.dropped_pre_session += 1  # the batch_seq hole becomes a visible gap
            ctx.pre_session.append(draft)

    def _drain(self, ctx: BootContext) -> None:
        while ctx.session is not None and ctx.pre_session:
            self._put_batch(ctx, ctx.pre_session.popleft())

    def _put_batch(self, ctx: BootContext, draft: BatchDraft) -> None:
        payload = to_spike_batch(
            draft, device_id=self._cfg.device.id, session_id=ctx.session.session_id, epoch=ctx.session.epoch,
            encoder_hash=self._cfg.session.encoder_hash,
        )  # fmt: skip
        self._enqueue(payload["request_id"], payload, "batch")

    def _enqueue(self, request_id: str, payload: dict, kind: str) -> None:
        result = self._state.put(request_id, payload, kind)
        if result.dropped:
            log.error("outbox full: dropped %d oldest entries (%s ...)", len(result.dropped), result.dropped[0])

    def _close_current(self) -> None:
        """The current boot is over (reset, disconnect, stop). Flush it and hand it to finalisation."""
        ctx, self._current = self._current, None
        if ctx is None:
            return
        if ctx.assembler is not None:
            self._route(ctx, ctx.assembler.flush())
        ctx.closed = True
        if ctx.create is None and ctx.session is None:
            return  # no frame ever arrived: nothing to create or stop
        self._closed.append(ctx)
        if len(self._closed) > MAX_CLOSED_CONTEXTS:
            lost = self._closed.pop(0)
            lost.cancelled = True
            with self._lock:
                if lost in self._create_queue:
                    self._create_queue.remove(lost)
            self._counters.dropped_pre_session += len(lost.pre_session)
            log.error("dropping boot %s: its session was never created", lost.boot_id)
        self._maintain()

    def _maintain(self) -> None:
        """Finalise closed contexts whose session exists, and hand buffered batches to the current one."""
        if self._current is not None:
            self._drain(self._current)
        for ctx in list(self._closed):
            if ctx.session is None:
                continue
            self._drain(ctx)
            self._enqueue_stop(ctx.session)
            self._closed.remove(ctx)

    def _enqueue_stop(self, ref: SessionRef) -> None:
        body = {
            "schema_version": SCHEMA_VERSION, "request_id": _rid("stop", ref.session_id),
            "device_id": self._cfg.device.id, "session_id": ref.session_id, "epoch": ref.epoch,
        }  # fmt: skip
        self._enqueue(body["request_id"], body, "stop")
        with self._lock:
            self._unstopped[ref.session_id] = body["request_id"]
        self._state.kv_delete("session")

    def _session_ref(self) -> SessionRef | None:
        ctx = self._current
        return ctx.session if ctx is not None else None

    def _recover_startup(self) -> None:
        """A session left running by a previous process is stopped; commands it interrupted are failed."""
        raw = self._state.kv_get("session")
        if raw is not None:
            try:
                data = json.loads(raw)
                self._enqueue_stop(SessionRef(data["session_id"], int(data["epoch"]), "demo", 0, 0))
                log.warning("stopping session %s left over from a previous run", data["session_id"])
            except (ValueError, KeyError, TypeError):
                self._state.kv_delete("session")
        self._handler.recover()

    def _shutdown(self, worker: threading.Thread, commands: threading.Thread) -> None:
        # Commands first: an in-flight capture finishes while its session still exists, and no new one starts.
        self._draining = True
        self._commands_stop.set()
        commands.join(timeout=self._cfg.limits.drain_s + 5)
        self._close_current()
        deadline = self._mono() + self._cfg.limits.drain_s
        while self._mono() < deadline:
            self._maintain()
            if not self._closed and self._state.stats().pending == 0:
                break
            time.sleep(0.02)
        for ctx in self._closed:
            self._counters.dropped_pre_session += len(ctx.pre_session)
        if self._closed:
            log.error("shutdown with %d boot(s) whose session was never created", len(self._closed))
        self._worker_stop.set()
        worker.join(timeout=self._cfg.backend.timeout_s + 2)

    # ----------------------------------------------------------- worker thread

    def _worker_loop(self) -> None:
        next_status = 0.0
        while not self._worker_stop.is_set():
            try:
                now = self._mono()
                self._create_sessions(now)
                self._pump_outbox(now)
                if now >= next_status:
                    next_status = now + self._cfg.limits.heartbeat_s
                    self._heartbeat()
            except Exception:  # the worker must survive anything; the next tick retries
                log.exception("worker error")
            self._worker_stop.wait(self._tick_s)

    def _stop_pending(self) -> bool:
        """True while an earlier session has not been stopped and delivered: the backend allows one."""
        with self._lock:
            items = list(self._unstopped.items())
        for session_id, stop_rid in items:
            state = self._state.state(stop_rid) if stop_rid else None
            if stop_rid is None or state == "pending":
                return True
            with self._lock:
                self._unstopped.pop(session_id, None)
        return False

    def _create_sessions(self, now: float) -> None:
        with self._lock:
            while self._create_queue and self._create_queue[0].cancelled:
                self._create_queue.popleft()
            ctx = self._create_queue[0] if self._create_queue else None
        if ctx is None or ctx.create is None or now < ctx.create.next_try or self._stop_pending():
            return
        request = ctx.create
        body = {
            "schema_version": SCHEMA_VERSION, "request_id": request.request_id, "device_id": self._cfg.device.id,
            "boot_id": ctx.boot_id, "mode": self._cfg.session.mode, "model_hash": self._cfg.session.model_hash,
            "encoder_hash": self._cfg.session.encoder_hash, "source_start_us": request.source_start_us,
        }  # fmt: skip
        result = self._api.create_session(body, self._cfg.backend.demo_scenario)
        if result.outcome is not Outcome.OK or not isinstance(result.body, dict):
            wait = SESSION_RETRY_PERMANENT_S if result.outcome is Outcome.PERMANENT else 0.0
            request.next_try = now + max(wait, request.backoff.next_delay(result.retry_after_s))
            request.last_error = result.code or result.detail
            log.warning("session create failed for boot %s: %s", ctx.boot_id, request.last_error)
            return
        limits = result.body.get("limits") if isinstance(result.body.get("limits"), dict) else {}
        ref = SessionRef(
            session_id=str(result.body["session_id"]), epoch=int(result.body["epoch"]), mode=self._cfg.session.mode,
            image_bytes=int(limits.get("image_bytes", DEFAULT_IMAGE_BYTES)),
            max_frames=int(limits.get("max_frames", DEFAULT_MAX_FRAMES)),
        )  # fmt: skip
        self._state.kv_set("session", json.dumps({"session_id": ref.session_id, "epoch": ref.epoch}))
        with self._lock:
            if self._create_queue and self._create_queue[0] is ctx:
                self._create_queue.popleft()
            self._unstopped[ref.session_id] = None  # running; its stop is enqueued when the boot ends
            ctx.session = ref
        log.info("session %s created for boot %s", ref.session_id, ctx.boot_id)
        if ctx.cancelled:  # abandoned while the request was in flight: do not leave it running
            self._enqueue_stop(ref)

    def _pump_outbox(self, now: float) -> None:
        if now < self._retry_at:
            return
        for entry in self._state.pending(20):
            result = self._deliver(entry)
            if result.outcome is Outcome.OK:
                self._state.mark_sent(entry.request_id)
                self._backoff.reset()
            elif result.outcome is Outcome.PERMANENT:
                self._state.mark_dead(entry.request_id, f"{result.status} {result.code or ''}".strip())
                log.error("dead-lettered %s %s: HTTP %s %s", entry.kind, entry.request_id, result.status, result.code)
            else:
                self._state.record_failure(entry.request_id, result.code or result.detail)
                self._retry_at = now + self._backoff.next_delay(result.retry_after_s)
                return  # keep order: nothing after a failed entry is sent first

    def _deliver(self, entry: Entry) -> Result:
        payload = entry.payload
        if entry.kind == "batch":
            return self._api.post_batch(payload["session_id"], payload)
        if entry.kind == "ack":
            return self._api.ack_command(payload["command_id"], payload)
        if entry.kind == "stop":
            return self._api.stop_session(payload["session_id"], payload)
        return Result(Outcome.PERMANENT, None, "UNKNOWN_KIND", None, None, f"unknown kind {entry.kind}")

    def _command_loop(self) -> None:
        while not self._commands_stop.is_set():
            try:
                self._poll_commands()
            except Exception:  # a bad command must never end the loop
                log.exception("command loop error")
            self._commands_stop.wait(self._cfg.limits.command_poll_s)

    def _poll_commands(self) -> None:
        if self._session_ref() is None:
            return
        result = self._api.poll_commands(self._cfg.device.id)
        if result.outcome is Outcome.OK and isinstance(result.body, dict):
            for item in result.body.get("items") or []:
                if self._commands_stop.is_set():
                    return  # shutting down: leave the rest to expire rather than fail them
                self._handler.handle(item)

    def _heartbeat(self) -> None:
        body = self.status()
        if self._api.post_status(self._cfg.device.id, body).outcome is not Outcome.OK:
            self._counters.status_failures += 1

    def status(self) -> dict:
        """The DeviceStatus payload for the current moment."""
        ctx, c = self._current, self._counters
        stats = self._state.stats()
        session = ctx.session if ctx else None
        now = self._mono()
        if self._draining:
            state, detail = "stopping", None
        elif ctx is not None and ctx.fault:
            state, detail = "error", ctx.fault
        elif not self._connected:
            state, detail = "reconnecting", None
        elif self._stalled:
            state, detail = "stalled", None
        elif session is not None and c.frames:
            state, detail = "running", None
        else:
            state, detail = "starting", ctx.create.last_error if ctx and ctx.create else None
        return {
            "schema_version": SCHEMA_VERSION,
            "request_id": f"status-{uuid.uuid4().hex[:16]}",
            "device_id": self._cfg.device.id,
            "reported_at": _utc_z(self._utc_now()),
            "agent_version": self._version[:64],
            "input_kind": self._cfg.device.input_kind,
            "state": state,
            "detail": detail[:200] if detail else None,
            "boot_id": ctx.boot_id if ctx else None,
            "session_id": session.session_id if session else None,
            "epoch": session.epoch if session else None,
            "uptime_s": int(now - self._started),
            "serial": {
                "connected": self._connected, "frames": c.frames, "gaps": c.gaps, "rejected": c.rejected,
                "stalls": c.stalls, "reconnects": c.reconnects,
                "last_frame_age_ms": None if self._last_frame is None else int((now - self._last_frame) * 1000),
            },
            "camera": {"configured": self._camera is not None},
            "outbox": {"pending": stats.pending, "dead": stats.dead, "dropped_total": stats.dropped_total},
        }  # fmt: skip


# ------------------------------------------------------------------ entry point


def build_bridge(config: Config, *, camera=None, sink=None) -> Bridge:
    token = read_token(config)
    os.makedirs(config.state_dir, mode=0o700, exist_ok=True)
    state = Outbox(os.path.join(config.state_dir, "outbox.db"), max_pending=config.limits.max_pending)
    api = ApiClient(
        UrllibTransport(config.backend.url, token=token, ca_file=config.backend.ca_file),
        timeout_s=config.backend.timeout_s,
    )
    serial = config.serial
    if config.device.input_kind == "replay":

        def source_factory() -> ByteSource:
            with open(serial.replay_file, "rb") as handle:
                return ReplaySource(handle.read())
    else:

        def source_factory() -> ByteSource:
            return SerialPortSource(serial.path, serial.baud)

    if sink is None and config.images.local_dir:
        from rpi_agents.agent.sinks import LocalDirSink

        sink = LocalDirSink(config.images.local_dir, config.images.keep)
    if camera is None and config.camera.serial:
        from rpi_agents.agent.camera import CameraConfig, UvcCamera

        camera = UvcCamera(CameraConfig(serial=config.camera.serial))
    return Bridge(
        config, api=api, state=state, source_factory=source_factory, camera=camera, sink=sink,
        agent_version=os.environ.get("SNN_EDGE_VERSION", "dev"),
    )  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SNN edge bridge: Uno serial stream to the backend")
    parser.add_argument("--config", required=True)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, stream=sys.stderr, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        bridge = build_bridge(load_config(args.config))
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())
    log.info("bridge starting")
    code = bridge.run(stop)
    log.info("bridge stopped")
    return code


if __name__ == "__main__":
    sys.exit(main())
