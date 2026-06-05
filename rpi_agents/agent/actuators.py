"""Actuators: red LED and piezo buzzer control (P1).

Top-level imports remain hardware-free; gpiozero is imported lazily inside functions.
"""

from agent import config, gpio

_led = None
_buzzer = None


def _devices():
    """Return (LED, Buzzer) singletons, initialising GPIO on first call."""
    global _led, _buzzer
    if _led is None or _buzzer is None:
        gpio.configure_gpio()
        from gpiozero import LED, Buzzer  # type: ignore[import-untyped]
        led = LED(config.LED_PIN)
        try:
            buzzer = Buzzer(config.BUZZER_PIN)
        except BaseException:
            led.close()
            raise
        _led, _buzzer = led, buzzer
    return _led, _buzzer


def self_test() -> None:
    """Blink LED 3× and beep buzzer 2× to verify hardware on bring-up."""
    import time
    led, buzzer = _devices()
    for _ in range(3):
        led.on()
        time.sleep(0.1)
        led.off()
        time.sleep(0.1)
    for _ in range(2):
        buzzer.on()
        time.sleep(0.1)
        buzzer.off()
        time.sleep(0.1)


def alarm_on() -> None:
    """Activate alarm: turn LED and buzzer on."""
    led, buzzer = _devices()
    led.on()
    buzzer.on()


def alarm_off() -> None:
    """Deactivate alarm: turn LED and buzzer off."""
    led, buzzer = _devices()
    led.off()
    buzzer.off()


def close() -> None:
    """Release GPIO devices and reset singletons (for tests and shutdown)."""
    global _led, _buzzer
    if _led is not None:
        _led.close()
        _led = None
    if _buzzer is not None:
        _buzzer.close()
        _buzzer = None


if __name__ == "__main__":
    self_test()
