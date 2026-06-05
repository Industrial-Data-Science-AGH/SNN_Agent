"""Unit tests for agent.gpio: backend setup and deny-list assertion."""

from gpiozero import Device
from gpiozero.pins.mock import MockFactory

from agent import gpio


def test_configure_gpio_returns_mock_factory_name():
    # conftest autouse fixture already set Device.pin_factory = MockFactory()
    name = gpio.configure_gpio()
    assert name == "MockFactory"


def test_configure_gpio_is_idempotent():
    factory_before = Device.pin_factory
    gpio.configure_gpio()
    gpio.configure_gpio()
    assert Device.pin_factory is factory_before


def test_assert_supported_backend_passes_with_mock():
    # MockFactory is not in DENY_BACKENDS — should not raise
    gpio.assert_supported_backend()


def test_assert_supported_backend_raises_for_deny_listed(monkeypatch):
    class RPiGPIOFactory:
        pass

    monkeypatch.setattr(Device, "pin_factory", RPiGPIOFactory())
    import pytest
    with pytest.raises(RuntimeError, match="deny-listed"):
        gpio.assert_supported_backend()


def test_configure_gpio_rejects_deny_listed_backend(monkeypatch):
    class RPiGPIOFactory:
        pass

    monkeypatch.setattr(Device, "pin_factory", RPiGPIOFactory())
    import pytest
    with pytest.raises(RuntimeError, match="deny-listed"):
        gpio.configure_gpio()
