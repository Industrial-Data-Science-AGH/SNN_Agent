"""Unit tests for agent.actuators: pin-state asserts via MockFactory."""

from unittest.mock import MagicMock

from gpiozero import Device

from agent import actuators, config


def test_alarm_on_sets_pins_high():
    actuators.alarm_on()
    assert Device.pin_factory.pin(config.LED_PIN).state == 1
    assert Device.pin_factory.pin(config.BUZZER_PIN).state == 1


def test_alarm_off_clears_pins():
    actuators.alarm_on()
    actuators.alarm_off()
    assert Device.pin_factory.pin(config.LED_PIN).state == 0
    assert Device.pin_factory.pin(config.BUZZER_PIN).state == 0


def test_self_test_runs_without_error(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    actuators.self_test()  # must not raise


def test_close_resets_singletons():
    actuators.alarm_on()
    actuators.close()
    # After close, _led and _buzzer are None; next call re-creates them.
    actuators.alarm_on()
    assert Device.pin_factory.pin(config.LED_PIN).state == 1


def test_devices_partial_init_does_not_leak_led(monkeypatch):
    """If Buzzer() raises during init, the LED must be closed, not leaked."""
    import pytest

    def boom(*_args, **_kwargs):
        raise RuntimeError("GPIO already in use")

    monkeypatch.setattr("gpiozero.Buzzer", boom)
    with pytest.raises(RuntimeError, match="GPIO already in use"):
        actuators._devices()
    # globals stayed clean; pin released
    assert actuators._led is None
    assert actuators._buzzer is None
    assert Device.pin_factory.pin(config.LED_PIN).state == 0


def test_buzzer_disabled_skips_buzzer_entirely(monkeypatch):
    """With BUZZER_ENABLED=False, no Buzzer is constructed and alarm_on only drives the LED."""
    monkeypatch.setattr(config, "BUZZER_ENABLED", False)
    led, buzzer = actuators._devices()
    assert buzzer is None

    actuators.alarm_on()
    assert Device.pin_factory.pin(config.LED_PIN).state == 1

    actuators.alarm_off()
    assert Device.pin_factory.pin(config.LED_PIN).state == 0


def test_buzzer_disabled_self_test_skips_beep(monkeypatch):
    """self_test() must not touch the buzzer when BUZZER_ENABLED=False."""
    monkeypatch.setattr(config, "BUZZER_ENABLED", False)
    monkeypatch.setattr("time.sleep", lambda _: None)
    actuators.self_test()  # must not raise even with piezo unwired
    assert actuators._buzzer is None


def test_blink_ends_with_led_off(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    actuators.blink(1.0, 0.5)
    assert Device.pin_factory.pin(config.LED_PIN).state == 0


def test_blink_toggles_led_at_least_once(monkeypatch):
    """Spy on led.on()/led.off() to confirm the loop actually toggles, not just sets once."""
    monkeypatch.setattr("time.sleep", lambda _: None)
    actuators._devices()  # pre-create so we can spy on the real LED instance
    led, _ = actuators._devices()
    on_spy = MagicMock(side_effect=led.on)
    off_spy = MagicMock(side_effect=led.off)
    monkeypatch.setattr(led, "on", on_spy)
    monkeypatch.setattr(led, "off", off_spy)

    actuators.blink(1.0, 0.5)  # 2 half-cycles: on, off

    assert on_spy.call_count >= 1
    assert off_spy.call_count >= 1
