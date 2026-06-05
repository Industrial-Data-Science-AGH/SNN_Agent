"""GPIO backend configuration: pin-factory setup and backend assertion (P1).

Top-level imports are hardware-free; gpiozero is imported lazily inside functions.
"""

DENY_BACKENDS: frozenset[str] = frozenset({"RPiGPIOFactory"})


def configure_gpio() -> str:
    """Set up the gpiozero pin factory if not already configured.

    On a Raspberry Pi, installs LGPIOFactory (lgpio backend, RP1-compatible).
    In tests, a MockFactory pre-set by the conftest fixture is left untouched.
    Always asserts the active backend is not deny-listed before returning.

    Returns:
        Class name of the active pin factory, or 'none' if unavailable.
    """
    from gpiozero import Device  # type: ignore[import-untyped]

    if Device.pin_factory is None:
        try:
            from gpiozero.pins.lgpio import LGPIOFactory  # type: ignore[import-untyped]
            Device.pin_factory = LGPIOFactory()
        except ImportError:
            pass  # Mac / CI: caller or test conftest must provide a factory

    assert_supported_backend()
    return type(Device.pin_factory).__name__ if Device.pin_factory is not None else "none"


def assert_supported_backend() -> None:
    """Raise RuntimeError if the active factory class is in the deny list.

    gpiozero may silently pick RPiGPIOFactory on older Pi OS images; the RP1
    GPIO controller in Pi 5 is incompatible with that legacy backend.

    Raises:
        RuntimeError: If the active factory class is deny-listed.
    """
    from gpiozero import Device  # type: ignore[import-untyped]

    name = type(Device.pin_factory).__name__ if Device.pin_factory is not None else "none"
    if name in DENY_BACKENDS:
        raise RuntimeError(
            f"GPIO backend '{name}' is deny-listed (RP1-incompatible). "
            "Use lgpio: sudo apt install python3-lgpio"
        )
