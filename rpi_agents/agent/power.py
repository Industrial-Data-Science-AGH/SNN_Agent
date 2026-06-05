"""Power management: wake/sleep/cooldown (P1).

Stubs for hardware integration. Top-level imports remain hardware-free;
subprocess and os are imported lazily inside functions.
"""


def resleep() -> None:
    """Put RPi 5 back into low-power halt mode.

    Raises:
        NotImplementedError: Phase P1 implementation pending.
    """
    raise NotImplementedError("P1: power.resleep()")


def cooldown(seconds: float) -> None:
    """Sleep for cooldown period before re-arming.

    Args:
        seconds: Duration to sleep (seconds).

    Raises:
        NotImplementedError: Phase P1 implementation pending.
    """
    del seconds
    raise NotImplementedError("P1: power.cooldown()")
