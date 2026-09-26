import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from contracts.validation import fixture, validate
from rpi_agents.agent.alarm import AlarmUnavailable
from rpi_agents.agent.camera import CameraDisconnected, CameraError, CameraOversize
from rpi_agents.agent.commands import CommandHandler, SessionRef, SinkUnavailable, ack_request_id
from rpi_agents.agent.outbox import Outbox
from rpi_agents.agent.ports import CameraImage

NOW = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)
SESSION = SessionRef("s1", 1, "demo", image_bytes=1_048_576, max_frames=3)


def command(**changes):
    base = fixture("capture-command") | {
        "device_id": "snn-pi", "session_id": "s1", "epoch": 1, "command_id": "c1", "event_id": "e1",
        "issued_at": "2026-09-24T11:59:59Z", "expires_at": "2026-09-24T12:00:10Z",
    }  # fmt: skip
    return base | changes


class Clock:
    def __init__(self):
        self.wall, self.mono = NOW, 100.0

    def utc(self):
        return self.wall

    def monotonic(self):
        return self.mono


class Camera:
    def __init__(self, clock, *, seconds=0.0, error=None):
        self.clock, self.seconds, self.error, self.calls = clock, seconds, error, []

    def capture(self, *, max_bytes):
        self.calls.append(max_bytes)
        self.clock.mono += self.seconds
        if self.error:
            raise self.error
        return CameraImage(b"\xff\xd8jpeg\xff\xd9", "2026-09-24T12:00:01.000Z")


class Sink:
    def __init__(self, error=None):
        self.stored, self.error = [], error

    def store(self, *, event_id, command_id, index, jpeg, captured_at):
        if self.error:
            raise self.error
        self.stored.append((event_id, command_id, index, len(jpeg), captured_at))
        return f"img-{len(self.stored)}"


class Alarm:
    led_available = buzzer_available = True

    def __init__(self, error=None, off_error=None):
        self.applied, self.offs, self.error, self.off_error = [], 0, error, off_error

    def apply(self, *, duration_ms, led, buzzer):
        if self.error:
            raise self.error
        self.applied.append((duration_ms, led, buzzer))

    def off(self):
        self.offs += 1
        if self.off_error:
            raise self.off_error


def alarm_command(**changes):
    base = fixture("alarm-command") | {
        "device_id": "snn-pi", "session_id": "s1", "epoch": 1, "command_id": "a1", "event_id": "e1",
        "issued_at": "2026-09-24T11:59:59Z", "expires_at": "2026-09-24T12:00:10Z",
    }  # fmt: skip
    return base | changes


@pytest.fixture
def env(tmp_path):
    class Env:
        pass

    e = Env()
    e.clock, e.acks = Clock(), []
    e.state = Outbox(str(tmp_path / "state.db"))
    e.camera, e.sink, e.session, e.alarm = Camera(e.clock), Sink(), SESSION, Alarm()
    e.make = lambda **kw: CommandHandler(
        device_id="snn-pi", state=e.state, camera=kw.get("camera", e.camera), sink=kw.get("sink", e.sink),
        alarm=kw.get("alarm", e.alarm), session=lambda: e.session, enqueue_ack=kw.get("enqueue", e.acks.append),
        utc_now=e.clock.utc, monotonic=e.clock.monotonic,
    )  # fmt: skip
    e.handler = e.make()
    return e


def statuses(env):
    return [a["status"] for a in env.acks]


def test_a_valid_capture_is_accepted_then_completed_with_an_image(env):
    assert env.handler.handle(command(parameters={"frames": 1, "max_bytes": 500_000})) == "completed"
    assert statuses(env) == ["accepted", "completed"]
    done = env.acks[1]
    assert done["image_id"] == "img-1" and done["error_code"] is None
    assert done["completed_at"] == "2026-09-24T12:00:00.000Z" and env.acks[0]["completed_at"] is None
    assert env.camera.calls == [500_000] and env.sink.stored[0][:3] == ("e1", "c1", 0)
    for ack in env.acks:
        validate("CommandAck", ack)
    assert env.state.command_status("c1") == "completed" and env.state.kv_get("cmd:c1") is None


def test_several_frames_are_all_stored_and_the_first_image_is_reported(env):
    env.handler.handle(command(parameters={"frames": 3, "max_bytes": 100}))
    assert [s[2] for s in env.sink.stored] == [0, 1, 2] and env.acks[-1]["image_id"] == "img-1"


def test_the_backend_image_limit_caps_the_request(env):
    env.session = SessionRef("s1", 1, "demo", image_bytes=1000, max_frames=1)
    env.handler.handle(command(parameters={"frames": 3, "max_bytes": 500_000}))
    assert env.camera.calls == [1000] and len(env.sink.stored) == 1


def test_a_duplicate_command_does_nothing_even_after_a_restart(env):
    assert env.handler.handle(command()) == "completed"
    assert env.handler.handle(command()) == "duplicate"
    assert env.make().handle(command()) == "duplicate"  # a new handler over the same durable state
    assert len(env.camera.calls) == 1 and statuses(env) == ["accepted", "completed"]


def test_a_command_that_is_already_expired_is_never_executed(env):
    assert env.handler.handle(command(expires_at="2026-09-24T11:59:59.500Z")) == "expired:COMMAND_EXPIRED"
    assert statuses(env) == ["expired"] and env.acks[0]["error_code"] == "COMMAND_EXPIRED"
    assert env.camera.calls == [] and env.state.command_status("c1") == "expired"
    validate("CommandAck", env.acks[0])


def test_a_slow_capture_that_outlives_the_ttl_is_expired_and_its_image_dropped(env):
    env.camera.seconds = 11.0  # the command lives 10 s
    assert env.handler.handle(command()) == "expired:COMMAND_EXPIRED"
    assert statuses(env) == ["accepted", "expired"] and env.sink.stored == []


def test_the_deadline_is_monotonic_so_a_wall_clock_jump_cannot_extend_it(env):
    class JumpBack(Camera):
        def capture(self, *, max_bytes):
            env.clock.wall -= timedelta(hours=1)  # NTP step backwards during the capture
            return super().capture(max_bytes=max_bytes)

    env.camera = JumpBack(env.clock, seconds=11.0)
    env.handler = env.make(camera=env.camera)
    assert env.handler.handle(command()) == "expired:COMMAND_EXPIRED"


def test_an_unknown_command_type_is_refused_and_touches_nothing(env):
    reboot = command(type="reboot", command_id="r1")
    assert env.handler.handle(reboot) == "failed:UNSUPPORTED_COMMAND"
    assert statuses(env) == ["failed"] and env.camera.calls == [] and env.alarm.applied == []


@pytest.mark.parametrize(
    "kwargs,code",
    [({"camera": None}, "CAMERA_NOT_CONFIGURED"), ({"sink": None}, "IMAGE_SINK_UNAVAILABLE")],
)
def test_without_a_camera_or_a_place_to_store_the_image_nothing_is_photographed(env, kwargs, code):
    handler = env.make(**kwargs)
    assert handler.handle(command()) == f"failed:{code}"
    assert statuses(env) == ["failed"] and env.camera.calls == []


@pytest.mark.parametrize(
    "changes,session,code",
    [
        ({"session_id": "other"}, SESSION, "SESSION_MISMATCH"),
        ({"epoch": 2}, SESSION, "SESSION_MISMATCH"),
        ({}, None, "SESSION_MISMATCH"),
        ({"mode": "live"}, SESSION, "MODE_MISMATCH"),
        ({"mode": "demo"}, SessionRef("s1", 1, "live", 1000, 3), "MODE_MISMATCH"),
    ],
)
def test_a_command_for_another_session_or_mode_is_failed_not_executed(env, changes, session, code):
    env.session = session
    assert env.handler.handle(command(**changes)) == f"failed:{code}"
    assert env.camera.calls == [] and env.acks[0]["session_id"] == command(**changes)["session_id"]


def test_replay_sessions_accept_demo_commands(env):
    env.session = SessionRef("s1", 1, "replay", 1_048_576, 3)
    assert env.handler.handle(command()) == "completed"


@pytest.mark.parametrize(
    "changes",
    [
        {"parameters": {"frames": 4, "max_bytes": 100}},
        {"parameters": {"frames": 0, "max_bytes": 100}},
        {"parameters": {"frames": 1, "max_bytes": 0}},
        {"parameters": {"frames": 1, "max_bytes": 2_000_000}},
        {"parameters": {"frames": True, "max_bytes": 100}},
        {"parameters": {"frames": 1, "max_bytes": 100, "extra": 1}},
        {"parameters": "many"},
        {"issued_at": "2026-09-24T12:00:10Z"},
        {"expires_at": "not a time"},
        {"expires_at": "2026-09-24T12:00:10+00:00"},
        {"schema_version": "2.0"},
        {"event_id": "bad id"},
    ],
)
def test_malformed_parameters_and_times_fail_without_touching_the_camera(env, changes):
    assert env.handler.handle(command(**changes)) == "failed:INVALID_COMMAND"
    assert env.camera.calls == []


@pytest.mark.parametrize("raw", [None, "text", [], {}, command(command_id="bad id"), command(epoch=True),
                                 command(epoch=0), command(session_id=None)])  # fmt: skip
def test_commands_that_cannot_even_be_acknowledged_are_ignored_quietly(env, raw):
    assert env.handler.handle(raw) == "invalid" and env.acks == [] and env.camera.calls == []


def test_a_command_for_another_device_is_ignored_and_not_claimed(env):
    assert env.handler.handle(command(device_id="other-pi")) == "ignored"
    assert env.acks == [] and env.state.command_status("c1") is None


@pytest.mark.parametrize(
    "camera_error,sink_error,code",
    [
        (CameraDisconnected("gone"), None, "CAMERA_DISCONNECTED"),
        (CameraOversize("big"), None, "IMAGE_TOO_LARGE"),
        (CameraError("x"), None, "CAMERA_ERROR"),
        (None, SinkUnavailable("down"), "IMAGE_UPLOAD_FAILED"),
        (RuntimeError("bug"), None, "INTERNAL_ERROR"),
    ],
)
def test_every_failure_ends_in_one_terminal_ack_and_never_raises(env, camera_error, sink_error, code):
    handler = env.make(camera=Camera(env.clock, error=camera_error), sink=Sink(sink_error))
    assert handler.handle(command()) == f"failed:{code}"
    assert statuses(env) == ["accepted", "failed"] and env.acks[1]["error_code"] == code
    validate("CommandAck", env.acks[1])


def test_a_command_interrupted_by_a_crash_is_failed_after_restart_exactly_once(env):
    class Crash(Camera):
        def capture(self, *, max_bytes):
            raise KeyboardInterrupt  # a process kill: not an Exception, so nothing cleans up

    crashing = env.make(camera=Crash(env.clock))
    with pytest.raises(KeyboardInterrupt):
        crashing.handle(command())
    assert env.state.command_status("c1") == "accepted" and statuses(env) == ["accepted"]
    restarted = env.make()
    assert restarted.recover() == 1
    assert statuses(env) == ["accepted", "failed"] and env.acks[1]["error_code"] == "AGENT_RESTARTED"
    assert restarted.recover() == 0 and restarted.handle(command()) == "duplicate"
    validate("CommandAck", env.acks[1])


def test_a_valid_alarm_is_accepted_applied_once_and_completed(env):
    assert env.handler.handle(alarm_command()) == "completed"
    assert statuses(env) == ["accepted", "completed"] and env.alarm.applied == [(1000, True, True)]
    assert env.acks[1]["completed_at"] == "2026-09-24T12:00:00.000Z" and env.acks[1]["image_id"] is None
    for ack in env.acks:
        validate("CommandAck", ack)
    assert env.state.command_status("a1") == "completed" and env.camera.calls == []


def test_a_repeated_alarm_command_never_buzzes_twice_even_after_a_restart(env):
    assert env.handler.handle(alarm_command()) == "completed"
    assert env.handler.handle(alarm_command()) == "duplicate"
    assert env.make().handle(alarm_command()) == "duplicate"
    assert env.alarm.applied == [(1000, True, True)] and statuses(env) == ["accepted", "completed"]


def test_an_expired_alarm_is_never_switched_on(env):
    assert env.handler.handle(alarm_command(expires_at="2026-09-24T11:59:59.500Z")) == "expired:COMMAND_EXPIRED"
    assert env.alarm.applied == [] and statuses(env) == ["expired"]


def test_an_alarm_whose_ttl_runs_out_between_accept_and_apply_is_not_applied(env):
    def slow_ack(ack):  # the accepted ack takes 11 s to enqueue: the 10 s command dies meanwhile
        env.acks.append(ack)
        env.clock.mono += 11.0

    handler = env.make(enqueue=slow_ack)
    assert handler.handle(alarm_command()) == "expired:COMMAND_EXPIRED"
    assert env.alarm.applied == [] and env.acks[-1]["status"] == "expired"


@pytest.mark.parametrize(
    "kwargs,changes,code",
    [
        ({"alarm": None}, {}, "ALARM_NOT_CONFIGURED"),
        ({}, {"session_id": "other"}, "SESSION_MISMATCH"),
        ({}, {"mode": "live"}, "MODE_MISMATCH"),
        ({}, {"parameters": {"duration_ms": 0, "led": True, "buzzer": False}}, "INVALID_COMMAND"),
        ({}, {"parameters": {"duration_ms": 30001, "led": True, "buzzer": False}}, "INVALID_COMMAND"),
        ({}, {"parameters": {"duration_ms": True, "led": True, "buzzer": False}}, "INVALID_COMMAND"),
        ({}, {"parameters": {"duration_ms": "1000", "led": True, "buzzer": False}}, "INVALID_COMMAND"),
        ({}, {"parameters": {"duration_ms": 1000, "led": "yes", "buzzer": False}}, "INVALID_COMMAND"),
        ({}, {"parameters": {"duration_ms": 1000, "led": False, "buzzer": False}}, "INVALID_COMMAND"),
        ({}, {"parameters": {"duration_ms": 1000, "led": True, "buzzer": False, "extra": 1}}, "INVALID_COMMAND"),
        ({}, {"parameters": {"led": True, "buzzer": False}}, "INVALID_COMMAND"),
        ({}, {"policy_version": None}, "INVALID_COMMAND"),
        ({}, {"policy_version": "bad version"}, "INVALID_COMMAND"),
        ({}, {"expires_at": "2026-09-24T12:00:31Z"}, "INVALID_COMMAND"),  # 32 s TTL: over the contract limit
    ],
)
def test_an_alarm_that_is_wrong_in_any_way_is_failed_and_nothing_is_switched_on(env, kwargs, changes, code):
    handler = env.make(**kwargs)
    assert handler.handle(alarm_command(**changes)) == f"failed:{code}"
    assert env.alarm.applied == [] and statuses(env) == ["failed"]
    validate("CommandAck", env.acks[0])


@pytest.mark.parametrize(
    "led_ok,buzzer_ok,params",
    [(True, False, {"led": True, "buzzer": True}), (False, True, {"led": True, "buzzer": False}),
     (True, False, {"led": False, "buzzer": True})],
)  # fmt: skip
def test_asking_for_an_output_that_is_not_wired_fails_before_anything_is_energised(env, led_ok, buzzer_ok, params):
    partly = Alarm()
    partly.led_available, partly.buzzer_available = led_ok, buzzer_ok
    handler = env.make(alarm=partly)
    result = handler.handle(alarm_command(parameters={"duration_ms": 500, **params}))
    assert result == "failed:OUTPUT_NOT_CONFIGURED" and partly.applied == []


def test_a_failing_alarm_ends_in_one_failed_ack_and_the_outputs_are_switched_off(env):
    unavailable = Alarm(error=AlarmUnavailable("gpio"))
    assert env.make(alarm=unavailable).handle(alarm_command()) == "failed:ALARM_UNAVAILABLE"
    crashing = Alarm(error=RuntimeError("bug"), off_error=OSError("stuck"))  # even off() failing must not escape
    assert env.make(alarm=crashing).handle(alarm_command(command_id="a2")) == "failed:ALARM_FAILED"
    assert crashing.offs == 1 and unavailable.offs == 0
    assert [a["error_code"] for a in env.acks if a["status"] == "failed"] == ["ALARM_UNAVAILABLE", "ALARM_FAILED"]


def test_an_alarm_interrupted_by_a_crash_is_failed_after_restart_and_never_repeated(env):
    class Crash(Alarm):
        def apply(self, **kw):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        env.make(alarm=Crash()).handle(alarm_command())
    restarted = env.make()
    assert restarted.recover() == 1 and env.acks[-1]["error_code"] == "AGENT_RESTARTED"
    assert restarted.handle(alarm_command()) == "duplicate" and env.alarm.applied == []


def test_ack_request_ids_are_stable_unique_and_within_the_contract_pattern():
    assert ack_request_id("c1", "accepted") == ack_request_id("c1", "accepted") != ack_request_id("c1", "failed")
    long_id = ack_request_id("x" * 64, "completed")
    assert len(long_id) <= 64 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", long_id)


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.agent.commands"], check=True)
