import subprocess
import sys
import types
from unittest.mock import patch

import pytest

from rpi_agents.agent import gpio


def test_device_seams_have_no_optional_dependencies():
    # -S removes site-packages: this must work even on a lean edge install.
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import rpi_agents.agent.ports, rpi_agents.agent.gpio, rpi_agents.runtime.ports",
        ],
        check=True,
    )


def test_imported_gpio_guard():
    class Device:
        pin_factory = type("MockFactory", (), {})()

    fake = types.ModuleType("gpiozero")
    fake.Device = Device
    with patch.dict(sys.modules, {"gpiozero": fake}):
        assert gpio.configure_gpio() == "MockFactory"
        Device.pin_factory = type("RPiGPIOFactory", (), {})()
        with pytest.raises(RuntimeError, match="deny-listed"):
            gpio.configure_gpio()
