"""Default-value guards and env-override tests for agent.config.

Tests here assert that the constants introduced in the field-tuning plan
have the correct defaults and respond to env overrides via importlib.reload.
"""

import importlib

import pytest

from agent import config


def test_gemini_model_default() -> None:
    assert config.GEMINI_MODEL == "gemini-3.1-flash-lite"


def test_motion_threshold_default() -> None:
    assert config.PREFILTER_MOTION_THRESHOLD == pytest.approx(0.08)


def test_capture_interval_default() -> None:
    assert config.CAPTURE_INTERVAL_S == pytest.approx(0.1)


def test_camera_usb_drain_default() -> None:
    assert config.CAMERA_USB_DRAIN_FRAMES == 4


def test_gemini_model_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """SNN_AGENT_GEMINI_MODEL env var is picked up on import/reload."""
    monkeypatch.setenv("SNN_AGENT_GEMINI_MODEL", "gemini-2.5-flash")
    importlib.reload(config)
    try:
        assert config.GEMINI_MODEL == "gemini-2.5-flash"
    finally:
        monkeypatch.delenv("SNN_AGENT_GEMINI_MODEL", raising=False)
        importlib.reload(config)
