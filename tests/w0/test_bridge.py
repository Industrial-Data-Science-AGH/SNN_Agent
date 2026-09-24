import copy
import threading
import time

import pytest
from fastapi.testclient import TestClient

from contracts.validation import fixture, validate
from rpi_agents.agent.api import ApiClient, Response, TransportError
from rpi_agents.agent.bridge import Bridge
from rpi_agents.agent.config import parse_config
from rpi_agents.agent.outbox import Outbox
from rpi_agents.agent.ports import CameraImage
from rpi_agents.agent.sources import SourceDisconnected, SourceEnded
from rpi_agents.agent.synthetic import scenario
from rpi_agents.cloud.app.mock_api import create_app

CREATE = fixture("session-create")
CHANNELS = [c["channel"] for c in fixture("model-manifest")["encoder_profile"]["channel_map"]]


class ClientTransport:
    """Transport into the real mock API, with switches to simulate an outage or a forced answer."""

    def __init__(self, client):
        self.client, self.down, self.overrides, self.log = client, False, [], []

    def request(self, method, path, body=None, *, headers=None, timeout_s=5.0):
        if self.down:
            raise TransportError("down")
        for match, response in self.overrides:
            if match(method, path, body):
                return response
        r = self.client.request(method, path, json=body, headers=headers)
        data = r.json() if r.content else None
        self.log.append((method, path, body, r.status_code, data))
        return Response(r.status_code, data if isinstance(data, dict) else None)


class Data:
    """Finite byte source that then ends ("end") or reports an unplugged cable ("disconnect")."""

    def __init__(self, data, then="end", chunk=64):
        self.data, self.then, self.chunk, self.pos = data, then, chunk, 0

    def read(self, max_bytes, timeout_s):
        if self.pos >= len(self.data):
            if self.then == "end":
                raise SourceEnded
            raise SourceDisconnected("unplugged")
        piece = self.data[self.pos : self.pos + self.chunk]
        self.pos += len(piece)
        return piece

    def close(self):
        pass


class Hold(Data):
    """Delivers its data, then stays open and silent until the bridge is stopped."""

    def read(self, max_bytes, timeout_s):
        if self.pos >= len(self.data):
            time.sleep(0.02)
            return b""
        return super().read(max_bytes, timeout_s)


class Rig:
    def __init__(self, tmp_path, client, **config):
        data = {
            "device": {"id": "demo-pi", "input_kind": "replay"},
            "backend": {"url": "http://127.0.0.1:8000", "demo_scenario": "silence", "timeout_s": 5},
            "session": {"mode": "demo", "model_hash": CREATE["model_hash"], "encoder_hash": CREATE["encoder_hash"]},
            "serial": {"replay_file": "/unused", "channels": CHANNELS, "stall_s": 5.0},
            "state": {"dir": str(tmp_path)},
            "limits": {"heartbeat_s": 0.05, "command_poll_s": 0.05, "drain_s": 5.0},
        }
        for dotted, value in config.items():
            section, key = dotted.split("__")
            data[section][key] = value
        self.config, self.client = parse_config(copy.deepcopy(data)), client
        self.store = client.app.state.demo_store
        self.transport = ClientTransport(client)
        self.outbox = Outbox(str(tmp_path / "outbox.db"), max_pending=self.config.limits.max_pending)
        self.bridge = None

    def make(self, *sources, camera=None, sink=None, **kw):
        queue = list(sources)

        def factory():
            if not queue:
                raise SourceDisconnected("no more sources")
            return queue.pop(0)

        self.bridge = Bridge(
            self.config, api=ApiClient(self.transport), state=self.outbox, source_factory=factory,
            camera=camera, sink=sink, worker_tick_s=0.02, **kw,
        )  # fmt: skip
        return self.bridge

    def start(self):
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.bridge.run, args=(self.stop,), daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        self.thread.join(timeout=15)
        assert not self.thread.is_alive(), "bridge did not stop"

    def sessions(self):
        return list(self.store.sessions.values())

    def batch_acks(self):
        return [r[4] for r in self.transport.log if r[1].endswith("/batches") and r[3] == 200]

    def gap_reasons(self):
        return [g["reason"] for ack in self.batch_acks() for g in ack["gaps"]]


def wait_for(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def make_rig(tmp_path):
    with TestClient(create_app(), base_url="http://127.0.0.1") as client:
        yield lambda **config: Rig(tmp_path, client, **config)


def test_a_replayed_glass_stream_reaches_the_backend_complete_and_the_session_is_stopped(make_rig):
    rig = make_rig()
    assert rig.make(Data(scenario("glass", 200))).run(threading.Event()) == 0
    (s,) = rig.sessions()
    assert (s["state"], s["received_seq"], s["count"]) == ("stopped", 7, 8)
    stats = rig.outbox.stats()
    assert (stats.pending, stats.dead, stats.dropped_total) == (0, 0, 0)
    assert all(ack["gaps"] == [] for ack in rig.batch_acks())
    assert (rig.bridge.counters.boots, rig.bridge.counters.frames) == (1, 200)


def test_the_fragment_of_a_line_at_connect_time_is_not_counted_as_an_error(make_rig):
    rig = make_rig()
    rig.make(Data(b"0,0,0*FD\r\n$F,7,70000,192,00,0,0*00\n" + scenario("glass", 100))).run(threading.Event())
    assert rig.bridge.counters.rejected == 0 and rig.sessions()[0]["received_seq"] == 3


def test_corrupt_lines_after_the_boot_line_are_counted_and_become_gaps(make_rig):
    rig = make_rig()
    rig.make(Data(scenario("corrupt", 200))).run(threading.Event())
    assert rig.bridge.counters.rejected == 4 and rig.bridge.counters.gaps == 3
    assert "missing_batch" in rig.gap_reasons()


def test_serial_loss_and_merged_frames_are_visible_to_the_backend(make_rig):
    rig = make_rig()
    rig.make(Data(scenario("lost", 200))).run(threading.Event())
    assert rig.gap_reasons() == ["missing_batch"] and rig.bridge.counters.gaps == 1
    rig = make_rig()
    rig.make(Data(scenario("late", 200))).run(threading.Event())
    assert "dropped_events" in rig.gap_reasons()


def test_a_backend_outage_loses_nothing_and_delivery_stays_in_order(make_rig):
    rig = make_rig()
    rig.transport.down = True
    threading.Timer(0.5, lambda: setattr(rig.transport, "down", False)).start()
    rig.make(Data(scenario("glass", 200))).run(threading.Event())
    sequence = [b[2]["batch_seq"] for b in rig.transport.log if b[1].endswith("/batches")]
    assert sequence == list(range(8))
    (s,) = rig.sessions()
    assert (s["state"], s["received_seq"]) == ("stopped", 7) and rig.outbox.stats().dead == 0
    assert rig.gap_reasons() == []


def test_a_reset_stops_the_old_session_before_the_new_one_is_created(make_rig):
    rig = make_rig()
    rig.make(Data(scenario("reset", 100))).run(threading.Event())
    first, second = rig.sessions()
    assert (first["state"], second["state"]) == ("stopped", "stopped")
    assert (first["received_seq"], second["received_seq"]) == (3, 3)
    steps = [(r[0], r[1].rsplit("/", 1)[-1]) for r in rig.transport.log if r[3] in (200, 201)]
    stop_at = next(i for i, s in enumerate(steps) if s[1] == "stop")
    creates = [i for i, s in enumerate(steps) if s == ("POST", "sessions?scenario=silence")]
    assert len(creates) == 2 and creates[0] < stop_at < creates[1]
    assert rig.bridge.counters.boots == 2


def test_a_disconnect_starts_a_new_boot_and_a_new_session(make_rig):
    rig = make_rig()
    rig.make(Data(scenario("glass", 100), then="disconnect"), Data(scenario("glass", 100))).run(threading.Event())
    first, second = rig.sessions()
    assert (first["state"], second["state"], second["received_seq"]) == ("stopped", "stopped", 3)
    assert rig.bridge.counters.reconnects == 1 and rig.bridge.counters.boots == 2


def test_a_channel_map_that_does_not_match_the_firmware_is_refused_not_guessed(make_rig):
    rig = make_rig(serial__channels=CHANNELS[:6])
    rig.make(Hold(scenario("glass", 60)))
    rig.start()
    try:
        assert wait_for(lambda: rig.store.status.get("demo-pi", {}).get("state") == "error")
        status = rig.store.status["demo-pi"]
        assert "channel map mismatch" in status["detail"] and status["session_id"] is None
        validate("DeviceStatus", status)
    finally:
        rig.finish()
    assert rig.sessions() == [] and rig.outbox.stats().pending == 0


def test_frames_without_a_boot_line_make_the_bridge_ask_the_device_to_reprint_it(make_rig):
    class Writable(Hold):
        written = []

        def write(self, data):
            self.written.append(data)

    rig = make_rig()
    source = Writable(b"".join(scenario("silence", 40).splitlines(keepends=True)[1:]))  # no $B line
    rig.make(source)
    rig.start()
    try:
        assert wait_for(lambda: source.written)
    finally:
        rig.finish()
    assert source.written == [b"I"] and rig.sessions() == []  # rate limited, and nothing sent without a boot


def test_an_unexpected_firmware_build_is_refused(make_rig):
    rig = make_rig(serial__expected_build_id="DEADBEEF")
    rig.make(Data(scenario("glass", 60))).run(threading.Event())
    assert rig.sessions() == [] and rig.bridge.counters.frames == 0


def test_buffer_overflow_before_the_session_exists_is_counted_not_hidden(make_rig):
    rig = make_rig(limits__pre_session_batches=2, limits__drain_s=0.3)
    rig.transport.down = True
    rig.make(Data(scenario("glass", 200))).run(threading.Event())
    assert rig.bridge.counters.dropped_pre_session >= 6 and rig.sessions() == []
    assert rig.outbox.stats().pending == 0  # nothing was stamped with a session it does not have


def test_the_pre_session_buffer_keeps_the_newest_batches_and_the_loss_is_a_visible_gap(make_rig):
    rig = make_rig(limits__pre_session_batches=2, limits__drain_s=15.0)
    rig.transport.down = True
    threading.Timer(0.5, lambda: setattr(rig.transport, "down", False)).start()
    rig.make(Data(scenario("glass", 200))).run(threading.Event())
    sent = [b[2]["batch_seq"] for b in rig.transport.log if b[1].endswith("/batches")]
    assert sent == [6, 7] and rig.bridge.counters.dropped_pre_session == 6  # only the newest two survived
    assert rig.gap_reasons() == ["missing_batch"]  # and the six lost ones are reported, not hidden


def test_boots_abandoned_while_the_backend_is_down_never_leave_a_session_running(make_rig):
    rig = make_rig(limits__drain_s=15.0)
    rig.transport.down = True
    threading.Timer(0.6, lambda: setattr(rig.transport, "down", False)).start()
    rig.make(Data(b"".join(scenario("glass", 30) for _ in range(10)))).run(threading.Event())
    states = [s["state"] for s in rig.sessions()]
    assert len(states) == 8 and set(states) == {"stopped"}  # the two oldest boots were dropped, not orphaned
    assert rig.bridge.counters.dropped_pre_session >= 2 and rig.bridge.counters.boots == 10
    assert rig.outbox.stats().pending == 0


def test_a_permanently_rejected_batch_is_dead_lettered_and_does_not_block_the_rest(make_rig):
    rig = make_rig()
    rig.transport.overrides.append((
        lambda m, p, b: p.endswith("/batches") and b["batch_seq"] == 2,
        Response(409, {"error": {"code": "OUT_OF_ORDER", "message": "x"}}),
    ))  # fmt: skip
    rig.make(Data(scenario("glass", 200))).run(threading.Event())
    stats = rig.outbox.stats()
    assert (stats.pending, stats.dead) == (0, 1)
    (s,) = rig.sessions()
    assert s["received_seq"] == 7 and "missing_batch" in rig.gap_reasons()  # the loss is explicit


def test_a_failing_heartbeat_endpoint_never_blocks_data(make_rig):
    rig = make_rig()
    rig.transport.overrides.append((lambda m, p, b: p.endswith("/status"), Response(500, None)))
    rig.make(Data(scenario("glass", 200))).run(threading.Event())
    assert rig.sessions()[0]["received_seq"] == 7 and rig.bridge.counters.status_failures >= 1


class Camera:
    def __init__(self):
        self.calls = []

    def capture(self, *, max_bytes):
        self.calls.append(max_bytes)
        return CameraImage(b"\xff\xd8x\xff\xd9", "2026-09-24T12:00:01.000Z")


class Sink:
    def __init__(self):
        self.stored = []

    def store(self, *, event_id, command_id, index, jpeg, captured_at):
        self.stored.append((event_id, command_id, index))
        return f"img-{len(self.stored)}"


def acks(rig):
    return [a["status"] for a in rig.store.acks.values()]


def test_a_trigger_becomes_exactly_one_capture_that_is_stored_and_acknowledged(make_rig):
    rig = make_rig(backend__demo_scenario="trigger")
    camera, sink = Camera(), Sink()
    rig.make(Hold(scenario("glass", 200)), camera=camera, sink=sink)
    rig.start()
    try:
        assert wait_for(lambda: "completed" in acks(rig))
        time.sleep(0.3)  # several more polls: the command must not be repeated
    finally:
        rig.finish()
    (command,) = rig.store.commands.values()
    (ack,) = rig.store.acks.values()
    assert ack["status"] == "completed" and ack["image_id"] == "img-1"
    assert sink.stored == [(command["event_id"], command["command_id"], 0)]
    assert len(camera.calls) == 1 and rig.outbox.stats().dead == 0


def test_a_slow_capture_never_stalls_delivery_and_runs_on_its_own_thread(make_rig):
    release, entered = threading.Event(), threading.Event()
    threads = []

    class Blocking(Camera):
        def capture(self, *, max_bytes):
            threads.append(threading.current_thread().name)
            entered.set()
            assert release.wait(10)
            return super().capture(max_bytes=max_bytes)

    rig = make_rig(backend__demo_scenario="trigger")
    rig.make(Hold(scenario("glass", 200)), camera=Blocking(), sink=Sink())
    rig.start()
    try:
        assert entered.wait(10)  # a capture is now blocked, as a real 1 s photo would be
        assert wait_for(lambda: rig.outbox.stats().pending == 0)  # yet the accepted ack and all batches went out
        assert rig.store.acks and [a["status"] for a in rig.store.acks.values()] == ["accepted"]
        release.set()
        assert wait_for(lambda: "completed" in acks(rig))
    finally:
        release.set()
        rig.finish()
    assert threads == ["snn-edge-commands"] and rig.outbox.stats().dead == 0


def test_shutting_down_lets_an_inflight_capture_finish_and_fails_no_command(make_rig):
    entered = threading.Event()

    class Slow(Camera):
        def capture(self, *, max_bytes):
            entered.set()
            time.sleep(0.4)
            return super().capture(max_bytes=max_bytes)

    rig = make_rig(backend__demo_scenario="trigger")
    rig.make(Hold(scenario("glass", 200)), camera=Slow(), sink=Sink())
    rig.start()
    try:
        assert entered.wait(10)
    finally:
        rig.finish()  # SIGTERM-style stop while the capture is still running
    assert set(acks(rig)) <= {"accepted", "completed"} and "completed" in acks(rig)
    assert rig.outbox.stats().dead == 0 and rig.sessions()[0]["state"] == "stopped"


def test_the_command_thread_is_stopped_before_the_boot_and_its_session_are_closed(make_rig):
    rig = make_rig()
    rig.make(Hold(scenario("glass", 60)))
    closes = []
    original = rig.bridge._close_current

    def spy():
        if rig.stop.is_set():  # only the closes caused by the stop request matter here
            closes.append(any(t.name == "snn-edge-commands" and t.is_alive() for t in threading.enumerate()))
        original()

    rig.bridge._close_current = spy
    rig.start()
    try:
        assert wait_for(lambda: rig.sessions())
    finally:
        rig.finish()
    assert closes and not any(closes)  # every stop-time close, or a capture could outlive its own session


def test_a_capture_without_a_camera_is_failed_honestly_and_nothing_is_taken(make_rig):
    rig = make_rig(backend__demo_scenario="trigger")
    rig.make(Hold(scenario("glass", 200)), camera=None, sink=Sink())
    rig.start()
    try:
        assert wait_for(lambda: "failed" in acks(rig))
    finally:
        rig.finish()
    (ack,) = rig.store.acks.values()
    assert (ack["status"], ack["error_code"]) == ("failed", "CAMERA_NOT_CONFIGURED")


def test_stopping_flushes_the_open_batch_and_stops_the_session(make_rig):
    rig = make_rig()
    rig.make(Hold(scenario("glass", 110)))  # 4 closed batches and 10 hops still open
    rig.start()
    try:
        assert wait_for(lambda: rig.sessions() and rig.sessions()[0]["received_seq"] == 3)
    finally:
        rig.finish()
    (s,) = rig.sessions()
    assert (s["state"], s["received_seq"]) == ("stopped", 4)  # the 5th, partial batch was flushed
    assert rig.outbox.stats().pending == 0


def test_a_session_left_running_by_a_previous_process_is_stopped_first(make_rig):
    rig = make_rig()
    old = rig.client.post("/v1/sessions?scenario=silence", json=CREATE, headers={"Idempotency-Key": "create-1"}).json()
    rig.outbox.kv_set("session", '{"session_id": "%s", "epoch": %d}' % (old["session_id"], old["epoch"]))
    rig.make(Data(scenario("glass", 100))).run(threading.Event())
    states = {s["session_id"]: s["state"] for s in rig.sessions()}
    assert len(states) == 2 and set(states.values()) == {"stopped"} and states[old["session_id"]] == "stopped"
    assert rig.outbox.kv_get("session") is None


def test_the_heartbeat_reports_the_real_state_and_validates(make_rig):
    rig = make_rig()
    rig.make(Hold(scenario("glass", 60)), camera=Camera(), sink=Sink())
    rig.start()
    try:
        assert wait_for(lambda: rig.store.status.get("demo-pi", {}).get("state") == "running")
        status = rig.store.status["demo-pi"]
    finally:
        rig.finish()
    validate("DeviceStatus", status)
    assert status["input_kind"] == "replay" and status["camera"]["configured"] is True
    assert status["serial"]["frames"] > 0 and status["session_id"] and status["boot_id"]
    validate("DeviceStatus", rig.bridge.status())  # also valid after the session ended


def test_the_bridge_and_all_its_modules_import_without_site_packages():
    import subprocess
    import sys

    code = "import rpi_agents.agent.bridge, rpi_agents.agent.commands, rpi_agents.agent.sinks, rpi_agents.agent.api"
    subprocess.run([sys.executable, "-S", "-c", code], check=True)
