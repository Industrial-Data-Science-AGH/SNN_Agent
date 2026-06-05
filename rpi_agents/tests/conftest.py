"""pytest configuration: make rpi_agents/ importable as 'agent.*'."""

import sys
from pathlib import Path

# Mirror test_forward.py:6 — prepend rpi_agents/ so `from agent import ...` resolves.
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from gpiozero import Device  # type: ignore[import-untyped]
from gpiozero.pins.mock import MockFactory  # type: ignore[import-untyped]


@pytest.fixture(autouse=True)
def reset_gpio():
    """Fresh MockFactory before each test; close actuator singletons after."""
    Device.pin_factory = MockFactory()
    yield
    from agent import actuators
    actuators.close()
    Device.pin_factory = MockFactory()
