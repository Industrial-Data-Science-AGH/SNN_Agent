"""Unit tests for agent.trigger: active-low GPIO simulation."""

import gpiozero

from agent import trigger


def _fake_button_cls(pressed: bool):
    """Return a Button stub class with the given is_pressed value."""

    class _Stub:
        is_pressed = pressed

        def __init__(self, _pin, pull_up=True):
            pass

        def close(self):
            pass

    return _Stub


def test_woken_by_trigger_when_pin_low(monkeypatch):
    monkeypatch.setattr(gpiozero, "Button", _fake_button_cls(pressed=True))
    ctx = trigger.read_wake_context()
    assert ctx.woken_by_trigger is True


def test_not_woken_by_trigger_when_pin_high(monkeypatch):
    monkeypatch.setattr(gpiozero, "Button", _fake_button_cls(pressed=False))
    ctx = trigger.read_wake_context()
    assert ctx.woken_by_trigger is False


def test_timestamps_are_positive_floats(monkeypatch):
    monkeypatch.setattr(gpiozero, "Button", _fake_button_cls(pressed=False))
    ctx = trigger.read_wake_context()
    assert isinstance(ctx.ts_monotonic, float) and ctx.ts_monotonic > 0
    assert isinstance(ctx.ts_wall, float) and ctx.ts_wall > 0
