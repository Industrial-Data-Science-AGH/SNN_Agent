"""SNN trigger detection: read GPIO wake signal (P1).

Top-level imports remain hardware-free; gpiozero is imported lazily inside functions.

In halt mode the wake-from-halt already occurred via GPIO3 before the Pi booted;
the Arduino latches WAKE_CONFIRM_PIN low to distinguish a real trigger from a manual boot.
"""

from agent import config, gpio
from agent.types import WakeContext


def read_wake_context() -> WakeContext:
    """Read wake context from GPIO and system clock.

    Captures monotonic and wall-clock timestamps, then reads WAKE_CONFIRM_PIN as
    active-low via gpiozero Button(pull_up=True). Pin low == Arduino trigger latched
    == woken_by_trigger=True.

    Returns:
        WakeContext with trigger status and timestamps.
    """
    import time
    from gpiozero import Button  # type: ignore[import-untyped]

    gpio.configure_gpio()
    ts_monotonic = time.monotonic()
    ts_wall = time.time()

    button = Button(config.WAKE_CONFIRM_PIN, pull_up=True)
    try:
        woken_by_trigger = button.is_pressed  # pressed == pin low == trigger latched
    finally:
        button.close()

    return WakeContext(
        woken_by_trigger=woken_by_trigger,
        ts_monotonic=ts_monotonic,
        ts_wall=ts_wall,
    )
