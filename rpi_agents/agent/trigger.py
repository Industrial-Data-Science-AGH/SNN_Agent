"""SNN trigger detection: read GPIO wake signal (P1).

Stubs for hardware integration. Top-level imports remain hardware-free;
lgpio is imported lazily inside functions.
"""

from agent.types import WakeContext


def read_wake_context() -> WakeContext:
    """Read wake context from GPIO and system clock.

    Captures monotonic and wall-clock timestamps, plus GPIO trigger state.

    Returns:
        WakeContext with trigger status and timestamps.

    Raises:
        NotImplementedError: Phase P1 implementation pending.
    """
    raise NotImplementedError("P1: trigger.read_wake_context()")
