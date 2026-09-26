import subprocess
import sys
import threading
import time
import types
from unittest.mock import patch

import pytest

from rpi_agents.agent.alarm import CONTRACT_MAX_MS, AlarmUnavailable, GpioAlarm


class Pin:
    def __init__(self, name, log, fail_off=False, fail_on=False):
        self.name, self.log, self.fail_off, self.fail_on, self.closed = name, log, fail_off, fail_on, False

    def on(self):
        if self.fail_on:
            raise OSError("boom")
        self.log.append((self.name, "on"))

    def off(self):
        self.log.append((self.name, "off"))
        if self.fail_off:
            raise OSError("stuck")

    def close(self):
        self.closed = True


class FakeTimer:
    """Records what would be scheduled instead of running it, so time is fully controlled."""

    created = []

    def __init__(self, interval, function):
        self.interval, self.function, self.daemon, self.cancelled = interval, function, False, False
        FakeTimer.created.append(self)

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


def alarm(log=None, *, buzzer=27, led=17, **kw):
    log = [] if log is None else log
    kw.setdefault("max_ms", 200)
    kw.setdefault("max_continuous_ms", 1000)
    made = kw.pop("made", {})

    def maker(name):
        return lambda pin: made.setdefault(name, Pin(name, log))

    return GpioAlarm(led_pin=led, buzzer_pin=buzzer, make_led=maker("led"), make_buzzer=maker("buzzer"), **kw), log, made


def wait_for(condition, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.01)
    return False


def state(log):
    """Last known state per output."""
    out = {}
    for name, action in log:
        out[name] = action
    return out


def test_nothing_is_created_or_energised_until_the_first_apply():
    a, log, made = alarm()
    assert log == [] and made == {}
    a.close()
    assert log == []  # closing something never used touches nothing


def test_outputs_are_forced_off_when_created_then_switched_on():
    a, log, _ = alarm()
    a.apply(duration_ms=100, led=True, buzzer=True)
    assert state(log) == {"led": "on", "buzzer": "on"}
    assert log.index(("led", "off")) < log.index(("led", "on"))  # initial off precedes on
    a.off()


def test_the_local_timer_switches_everything_off_without_any_further_call():
    a, log, _ = alarm()
    a.apply(duration_ms=80, led=True, buzzer=True)
    assert state(log) == {"led": "on", "buzzer": "on"}
    assert wait_for(lambda: state(log) == {"led": "off", "buzzer": "off"})  # nobody called off()


def test_the_local_maximum_beats_a_longer_request():
    FakeTimer.created = []
    a, log, _ = alarm(max_ms=100, max_continuous_ms=5000, timer=FakeTimer)
    a.apply(duration_ms=CONTRACT_MAX_MS, led=True, buzzer=False)
    assert [t.interval for t in FakeTimer.created] == [0.1]  # 30 s were asked for, the local limit granted 100 ms
    a.apply(duration_ms=20, led=True, buzzer=False)
    assert [round(t.interval, 3) for t in FakeTimer.created] == [0.1, 0.02]  # and a shorter request is honoured


def test_only_the_requested_outputs_are_energised_and_a_smaller_request_turns_the_other_off():
    a, log, _ = alarm()
    a.apply(duration_ms=500, led=True, buzzer=False)
    assert state(log) == {"led": "on"}
    a.apply(duration_ms=500, led=True, buzzer=True)
    assert state(log) == {"led": "on", "buzzer": "on"}
    a.apply(duration_ms=500, led=False, buzzer=True)
    assert state(log) == {"led": "off", "buzzer": "on"}
    a.off()


def test_an_unconfigured_output_is_refused_before_anything_is_energised():
    a, log, made = alarm(buzzer=None)
    assert a.led_available and not a.buzzer_available
    with pytest.raises(AlarmUnavailable, match="buzzer is not configured"):
        a.apply(duration_ms=100, led=True, buzzer=True)
    assert log == [] and made == {}


@pytest.mark.parametrize("duration", [0, -5, True, 1.5, "100"])
def test_bad_durations_and_empty_requests_are_rejected(duration):
    a, log, _ = alarm()
    with pytest.raises(ValueError):
        a.apply(duration_ms=duration, led=True, buzzer=False)
    assert log == []


def test_an_alarm_with_no_active_output_is_rejected():
    a, log, _ = alarm()
    with pytest.raises(ValueError, match="at least one"):
        a.apply(duration_ms=100, led=False, buzzer=False)
    assert log == []


def test_off_always_tries_every_output_even_when_one_fails():
    log = []
    made = {"led": Pin("led", log, fail_off=True), "buzzer": Pin("buzzer", log)}
    a, _, _ = alarm(log, made=made)
    made["led"].fail_off = False
    a.apply(duration_ms=500, led=True, buzzer=True)
    made["led"].fail_off = True
    log.clear()
    with pytest.raises(AlarmUnavailable, match="did not switch off"):
        a.off()
    assert ("buzzer", "off") in log and ("led", "off") in log  # the buzzer was switched off despite the LED failing


def test_a_failing_output_leaves_everything_off_and_reports_unavailable():
    log = []
    made = {"led": Pin("led", log), "buzzer": Pin("buzzer", log, fail_on=True)}
    a, _, _ = alarm(log, made=made)
    with pytest.raises(AlarmUnavailable, match="output failed"):
        a.apply(duration_ms=500, led=True, buzzer=True)
    assert state(log) == {"led": "off", "buzzer": "off"}


def test_repeated_applies_cannot_keep_the_alarm_on_beyond_the_continuous_limit():
    FakeTimer.created = []
    clock = [0.0]
    a, log, _ = alarm(max_ms=100, max_continuous_ms=200, monotonic=lambda: clock[0], timer=FakeTimer)
    a.apply(duration_ms=100, led=True, buzzer=False)  # t=0.00, 0 ms used
    clock[0] = 0.07
    a.apply(duration_ms=100, led=True, buzzer=False)  # 70 ms used, 130 left: the full 100 ms is granted
    clock[0] = 0.16
    a.apply(duration_ms=100, led=True, buzzer=False)  # 160 ms used: only the 40 ms that are left
    assert [round(t.interval, 3) for t in FakeTimer.created] == [0.1, 0.1, 0.04]
    assert [t.cancelled for t in FakeTimer.created] == [True, True, False]  # one live timer at a time
    clock[0] = 0.2
    with pytest.raises(AlarmUnavailable, match="continuous"):
        a.apply(duration_ms=100, led=True, buzzer=False)
    assert state(log) == {"led": "off"} and FakeTimer.created[-1].cancelled


def test_off_is_idempotent_cancels_the_timer_and_close_releases_the_pins():
    a, log, made = alarm()
    a.apply(duration_ms=500, led=True, buzzer=True)
    a.off()
    a.off()
    assert state(log) == {"led": "off", "buzzer": "off"}
    before = len(log)
    time.sleep(0.15)
    assert len(log) == before  # the cancelled timer did not fire again
    a.close()
    assert made["led"].closed and made["buzzer"].closed


def test_constructor_validation():
    with pytest.raises(ValueError, match="at least one"):
        GpioAlarm(led_pin=None, buzzer_pin=None)
    with pytest.raises(ValueError):
        GpioAlarm(led_pin=17, max_ms=0)
    with pytest.raises(ValueError):
        GpioAlarm(led_pin=17, max_ms=CONTRACT_MAX_MS + 1)
    with pytest.raises(ValueError):
        GpioAlarm(led_pin=17, max_ms=5000, max_continuous_ms=1000)


def test_the_default_outputs_use_gpiozero_through_the_guarded_backend():
    created = []

    class Device:
        pin_factory = type("MockFactory", (), {})()

    class LED:
        def __init__(self, pin):
            created.append(("LED", pin))

        def on(self): ...
        def off(self): ...
        def close(self): ...

    class Buzzer(LED):
        def __init__(self, pin):
            created.append(("Buzzer", pin))

    fake = types.ModuleType("gpiozero")
    fake.Device, fake.LED, fake.Buzzer = Device, LED, Buzzer
    with patch.dict(sys.modules, {"gpiozero": fake}):
        a = GpioAlarm(led_pin=17, buzzer_pin=27, max_ms=100)
        a.apply(duration_ms=50, led=True, buzzer=True)
        assert created == [("LED", 17), ("Buzzer", 27)]
        a.close()
        Device.pin_factory = type("RPiGPIOFactory", (), {})()  # the deny-listed legacy backend
        with pytest.raises(AlarmUnavailable):
            GpioAlarm(led_pin=17, max_ms=100).apply(duration_ms=50, led=True, buzzer=False)


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.agent.alarm"], check=True)


def test_concurrent_applies_and_offs_do_not_deadlock_or_leave_the_alarm_on():
    a, log, _ = alarm(max_ms=50, max_continuous_ms=5000)
    stop = threading.Event()

    def hammer(which):
        while not stop.is_set():
            try:
                a.apply(duration_ms=30, led=True, buzzer=bool(which)) if which != 2 else a.off()
            except AlarmUnavailable:
                pass

    threads = [threading.Thread(target=hammer, args=(i,)) for i in range(3)]
    [t.start() for t in threads]
    time.sleep(0.3)
    stop.set()
    [t.join(timeout=5) for t in threads]
    assert not any(t.is_alive() for t in threads)
    assert wait_for(lambda: set(state(log).values()) <= {"off"})
