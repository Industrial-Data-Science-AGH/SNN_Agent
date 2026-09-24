"""The whole chain, in one process, with the real code on both sides.

REAL: the edge bridge (serial protocol, batching, outbox, session handling, command handler, image upload,
alarm handler), the production HTTP API with its bearer-token authentication, sessions/ingest/commands/images,
the outbox publisher, the worker, the alarm policy. FAKED: the serial device (a replayed byte stream), the camera,
the GPIO alarm, the SNN runtime (a spike-count stand-in), the vision model, the mailbox. Nothing here says
anything about the real hardware, the real SNN or a real model.
"""

import hashlib
import threading
import time

import pytest
from fastapi.testclient import TestClient

from contracts.validation import fixture
from rpi_agents.agent.api import ApiClient, Response
from rpi_agents.agent.bridge import Bridge
from rpi_agents.agent.config import parse_config
from rpi_agents.agent.outbox import Outbox
from rpi_agents.agent.ports import CameraImage
from rpi_agents.agent.sinks import BackendImageSink
from rpi_agents.agent.synthetic import scenario
from rpi_agents.cloud.app.api import ApiSettings, Services, create_app
from rpi_agents.cloud.app.auth import OperatorAuth, hash_password, issue_device_token
from rpi_agents.cloud.app.policy import ARMED
from rpi_agents.cloud.app.records import CORE, T_DEVICE_COMMANDS, T_EVENTS
from rpi_agents.cloud.app.status import StatusService
from rpi_agents.cloud.app.vision import Observation, VisionUnavailable
from tests.w0.backend_env import Env, SpikeRuntime
from tests.w0.fakes import FakeNotifier, FakeVision, make_jpeg
from tests.w0.test_bridge import Hold, wait_for

CHANNELS = [c["channel"] for c in fixture("model-manifest")["encoder_profile"]["channel_map"]]
CREATE = fixture("session-create")
GLASS = Observation(True, True, "good", "Broken glass and a person near the window.")


class ApiTransport:
    """The bridge's Transport, wired straight into the production app (as a real device would be over HTTPS)."""

    def __init__(self, client, token):
        self.client, self.token, self.down = client, token, False

    def request(self, method, path, body=None, *, headers=None, timeout_s=5.0, raw=None):
        from rpi_agents.agent.api import TransportError

        if self.down:
            raise TransportError("down")
        headers = {"Authorization": f"Bearer {self.token}"} | dict(headers or {})
        kwargs = {"content": raw} if raw is not None else {"json": body} if body is not None else {}
        r = self.client.request(method, path, headers=headers, **kwargs)
        data = r.json() if r.content and r.headers.get("content-type", "").startswith("application/json") else None
        return Response(r.status_code, data if isinstance(data, dict) else None)


class FakeCamera:
    def __init__(self):
        self.calls = []

    def capture(self, *, max_bytes):
        self.calls.append(max_bytes)
        return CameraImage(make_jpeg(640, 480), "2026-09-24T12:00:03.000Z")


class FakeAlarm:
    led_available = buzzer_available = True

    def __init__(self):
        self.applied, self.offs = [], 0

    def initialize(self):
        pass

    def apply(self, *, duration_ms, led, buzzer):
        self.applied.append((duration_ms, led, buzzer))

    def off(self):
        self.offs += 1


class System:
    def __init__(self, tmp_path, *, policy=ARMED, vision=None, source=None, **backend):
        self.env = Env(policy_version=policy, recipients=("owner@example.com",), event_cooldown_s=0, **backend)
        env = self.env
        env.runtime_class = SpikeRuntime
        services = Services(env.sessions, env.ingest, env.commands, env.images, env.events, StatusService(env.ctx), env.ctx)
        operator = OperatorAuth(env.ctx, username="operator", password_hash=hash_password("pw-for-tests", log2_n=14))
        self.client = TestClient(create_app(services, operator, ApiSettings(trusted_proxies=0)), base_url="https://testserver")
        self.token = issue_device_token(env.ctx, "demo-pi")
        self.transport = ApiTransport(self.client, self.token)

        self.vision = vision or FakeVision([GLASS] * 5)
        self.notifier = FakeNotifier()
        self.worker = env.worker(self.vision, self.notifier, renew_interval_s=0.05)
        self.camera, self.alarm = FakeCamera(), FakeAlarm()

        config = parse_config({
            "device": {"id": "demo-pi", "input_kind": "replay"},
            "backend": {"url": "http://127.0.0.1:8000", "timeout_s": 5},
            "session": {"mode": "demo", "model_hash": CREATE["model_hash"], "encoder_hash": CREATE["encoder_hash"]},
            "serial": {"replay_file": "/unused", "channels": CHANNELS, "stall_s": 5.0},
            "images": {"upload": True}, "state": {"dir": str(tmp_path)},
            "limits": {"heartbeat_s": 0.2, "command_poll_s": 0.05, "drain_s": 5.0},
        })  # fmt: skip
        api = ApiClient(self.transport)
        self.outbox = Outbox(str(tmp_path / "edge.db"))
        queue = [source or Hold(scenario("glass", 200))]
        self.bridge = Bridge(
            config, api=api, state=self.outbox, source_factory=lambda: queue.pop(0), camera=self.camera,
            sink=BackendImageSink(api), alarm=self.alarm, utc_now=env.clock, worker_tick_s=0.02,
        )  # fmt: skip
        self.stop_edge, self.stop_worker = threading.Event(), threading.Event()

    def start(self):
        self.worker_thread = threading.Thread(target=self.worker.run_forever, args=(self.stop_worker, 0.02), daemon=True)
        self.edge_thread = threading.Thread(target=self.bridge.run, args=(self.stop_edge,), daemon=True)
        self.worker_thread.start()
        self.edge_thread.start()

    def stop(self):
        self.stop_edge.set()
        self.edge_thread.join(15)
        self.stop_worker.set()
        self.worker_thread.join(5)
        assert not self.edge_thread.is_alive() and not self.worker_thread.is_alive()

    def events(self):
        return [self.env.events.get(r.data["event_id"]) for r in self.env.storage.tables.query(T_EVENTS, CORE, rk_prefix="list:")]

    def command_status(self, type_):
        rows = self.env.storage.tables.query(T_DEVICE_COMMANDS, "demo-pi")
        return [(r.data["command"]["command_id"], r.data["status"]) for r in rows if r.data["command"]["type"] == type_]


@pytest.fixture
def system(tmp_path):
    made = []

    def build(**kw):
        directory = tmp_path / f"s{len(made)}"
        directory.mkdir()
        s = System(directory, **kw)
        made.append(s)
        return s

    yield build
    for s in made:
        s.stop_edge.set()
        s.stop_worker.set()


def test_glass_and_a_person_travel_the_whole_chain_from_serial_bytes_to_a_buzzer_and_an_email(system):
    s = system()
    s.start()
    try:
        assert wait_for(lambda: s.command_status("alarm") and s.command_status("alarm")[0][1] == "completed", 15)
        time.sleep(0.4)  # extra polls: the alarm command must not act twice
    finally:
        s.stop()
    (event,) = s.events()
    assert event["status"] == "alarm_confirmed" and [c["type"] for c in event["commands"]] == ["capture", "alarm"]
    assert event["vision"]["glass_visible"] is True and event["vision"]["image_id"] == f"{event['event_id']}-0"
    assert s.command_status("capture")[0][1] == "completed"  # the device took the photo and the backend stored it
    assert s.camera.calls == [1_048_576] and s.alarm.applied == [(10_000, True, True)]  # one photo, one alarm
    assert len(s.vision.calls) == 1 and s.vision.calls[0][:2] == b"\xff\xd8"  # the model was shown the stored photo
    (mail,) = s.notifier.sent
    assert (mail.status, mail.reason, mail.event_id) == ("alarm_confirmed", "GLASS_AND_PERSON", event["event_id"]) and mail.jpeg is not None
    assert hashlib.sha256(mail.jpeg).hexdigest() == hashlib.sha256(s.env.images.read(event["event_id"], 0)).hexdigest()
    (session,) = s.env.storage.tables.query("sessions", "demo-pi")
    assert session.data["state"] == "stopped" and s.outbox.stats().dead == 0  # a clean stop, nothing lost on the edge
    assert s.alarm.offs >= 1  # the bridge switched the outputs off when it stopped


def test_under_the_default_policy_the_same_scene_reaches_a_human_but_never_the_buzzer(system):
    s = system(policy="manual-review-only-v1")
    s.start()
    try:
        assert wait_for(lambda: s.events() and s.events()[0]["status"] == "review_required" and s.notifier.sent, 15)
        time.sleep(0.3)
    finally:
        s.stop()
    (event,) = s.events()
    assert [c["type"] for c in event["commands"]] == ["capture"] and s.command_status("alarm") == []
    assert s.alarm.applied == [] and s.notifier.sent[0].reason == "MANUAL_REVIEW_POLICY"


def test_a_vision_outage_ends_in_human_review_and_a_silent_buzzer(system):
    s = system(vision=FakeVision([VisionUnavailable("VISION_TIMEOUT")] * 3), vision_retry_delay_s=1, vision_visibility_s=2)
    s.start()
    try:
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline and not (s.events() and s.events()[0]["status"] == "review_required"):
            s.env.clock.advance(1.5)  # time passes for the queue's retry delay
            time.sleep(0.05)
    finally:
        s.stop()
    (event,) = s.events()
    assert (event["status"], event["vision"]["status"]) == ("review_required", "unavailable")
    assert s.alarm.applied == [] and s.notifier.sent and s.notifier.sent[0].reason == "VISION_UNAVAILABLE"


def test_a_backend_restart_in_the_middle_gives_a_new_epoch_and_the_chain_still_works_afterwards(system):
    from tests.w0.test_bridge import Phased

    lines = scenario("glass", 300).splitlines(keepends=True)  # one continuous stream: bursts at hops 100 and 200
    probe, rest = threading.Event(), threading.Event()
    s = system(source=Phased(b"".join(lines[:101]), [(probe, b"".join(lines[101:126])), (rest, b"".join(lines[126:]))]))
    s.start()
    sessions = lambda: sorted(s.env.storage.tables.query("sessions", "demo-pi"), key=lambda r: r.data["epoch"])  # noqa: E731
    try:
        assert wait_for(lambda: sessions() and sessions()[0].data["received_seq"] == 3, 10)
        s.env.ctx.runtimes.clear()  # the backend process restarts: every live runtime is gone
        probe.set()  # the next batch (with the first burst) is refused, which is how the edge finds out
        assert wait_for(lambda: len(sessions()) == 2, 10)
        rest.set()  # from here on the stream has a live session again; the burst at hop 200 must get through
        assert wait_for(lambda: s.command_status("alarm") and s.command_status("alarm")[0][1] == "completed", 20)
    finally:
        s.stop()
    assert [x.data["epoch"] for x in sessions()] == [1, 2] and sessions()[0].data["stop_reason"] == "backend_restarted"
    assert s.alarm.applied == [(10_000, True, True)]  # one alarm, from the burst that arrived after the new epoch began
    (event,) = s.events()
    assert event["epoch"] == 2 and event["status"] == "alarm_confirmed"


def test_the_edge_keeps_its_data_through_a_backend_outage_and_the_chain_completes_after_it(system):
    s = system(source=Hold(scenario("glass", 200)))
    s.transport.down = True
    threading.Timer(0.8, lambda: setattr(s.transport, "down", False)).start()
    s.start()
    try:
        assert wait_for(lambda: s.command_status("alarm") and s.command_status("alarm")[0][1] == "completed", 20)
    finally:
        s.stop()
    assert s.alarm.applied == [(10_000, True, True)] and s.outbox.stats().dead == 0
    (event,) = s.events()
    assert event["status"] == "alarm_confirmed"
