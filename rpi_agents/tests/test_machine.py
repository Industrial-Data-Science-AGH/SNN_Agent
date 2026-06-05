"""State-transition tests for agent.machine.run_cycle() (P4)."""

import json
import smtplib
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from agent import config, machine
from agent.types import Decision, PrefilterResult, VisionVerdict, WakeContext


# ── helpers ──────────────────────────────────────────────────────────────────

def _frames() -> np.ndarray:
    """Small BGR frame stack (4, 8, 8, 3)."""
    return np.zeros((4, 8, 8, 3), dtype=np.uint8)


def _wake() -> WakeContext:
    return WakeContext(woken_by_trigger=True, ts_monotonic=0.0, ts_wall=0.0)


def _pf_static() -> PrefilterResult:
    return PrefilterResult(motion=False, person=False, escalate=False, score=0.0)


def _pf_escalate() -> PrefilterResult:
    return PrefilterResult(motion=True, person=True, escalate=True, score=0.8)


def _clip_path() -> Path:
    return Path("/tmp/fake_clip.mp4")


# ── fixtures ─────────────────────────────────────────────────────────────────

def _patch_leaves(monkeypatch, *, pf, verdict=None, notify_raises=None):
    """Monkeypatch all leaf collaborators; return spy mocks."""
    frames = _frames()
    monkeypatch.setattr("agent.camera.capture", lambda **kw: frames)
    monkeypatch.setattr("agent.prefilter.run", lambda f: pf)
    monkeypatch.setattr("agent.notifier.save_clip", lambda f: _clip_path())

    vision_spy = MagicMock(return_value=verdict)
    monkeypatch.setattr("agent.vision.verdict", vision_spy)

    notify_spy = MagicMock(side_effect=notify_raises)
    monkeypatch.setattr("agent.notifier.notify", notify_spy)

    alarm_on_spy = MagicMock()
    monkeypatch.setattr("agent.actuators.alarm_on", alarm_on_spy)

    save_clip_spy = MagicMock(return_value=_clip_path())
    monkeypatch.setattr("agent.notifier.save_clip", save_clip_spy)

    return vision_spy, notify_spy, alarm_on_spy, save_clip_spy


# ── tests ─────────────────────────────────────────────────────────────────────

def test_static_scene_no_alarm_no_vision(monkeypatch):
    vision_spy, notify_spy, alarm_on_spy, save_clip_spy = _patch_leaves(
        monkeypatch, pf=_pf_static()
    )
    decision = machine.run_cycle(_wake())

    assert decision.alarm is False
    vision_spy.assert_not_called()
    alarm_on_spy.assert_not_called()
    notify_spy.assert_not_called()
    save_clip_spy.assert_called_once()


def test_escalate_vision_false_no_alarm(monkeypatch):
    v = VisionVerdict(is_intrusion=False, confidence=0.9, reason="headlights", source="gemini")
    vision_spy, notify_spy, alarm_on_spy, _ = _patch_leaves(
        monkeypatch, pf=_pf_escalate(), verdict=v
    )
    decision = machine.run_cycle(_wake())

    assert decision.alarm is False
    vision_spy.assert_called_once()
    alarm_on_spy.assert_not_called()
    notify_spy.assert_not_called()


def test_escalate_vision_true_alarms(monkeypatch):
    v = VisionVerdict(is_intrusion=True, confidence=0.95, reason="person at window", source="gemini")
    vision_spy, notify_spy, alarm_on_spy, save_clip_spy = _patch_leaves(
        monkeypatch, pf=_pf_escalate(), verdict=v
    )
    decision = machine.run_cycle(_wake())

    assert decision.alarm is True
    alarm_on_spy.assert_called_once()
    notify_spy.assert_called_once()
    save_clip_spy.assert_called_once()


def test_failsafe_verdict_alarms(monkeypatch):
    v = VisionVerdict(is_intrusion=True, confidence=1.0, reason="failsafe: TimeoutError", source="failsafe")
    _, _, alarm_on_spy, _ = _patch_leaves(monkeypatch, pf=_pf_escalate(), verdict=v)
    decision = machine.run_cycle(_wake())

    assert decision.alarm is True


def test_vision_raises_machine_failsafe_alarms(monkeypatch):
    _patch_leaves(monkeypatch, pf=_pf_escalate(), verdict=None)
    monkeypatch.setattr("agent.vision.verdict", MagicMock(side_effect=RuntimeError("boom")))

    alarm_on_spy = MagicMock()
    monkeypatch.setattr("agent.actuators.alarm_on", alarm_on_spy)

    decision = machine.run_cycle(_wake())

    assert decision.alarm is True
    assert decision.reason.startswith("machine-failsafe")
    alarm_on_spy.assert_called_once()


def test_notify_failure_does_not_block_alarm(monkeypatch):
    v = VisionVerdict(is_intrusion=True, confidence=0.95, reason="intruder", source="gemini")
    _, notify_spy, alarm_on_spy, _ = _patch_leaves(
        monkeypatch, pf=_pf_escalate(), verdict=v,
        notify_raises=smtplib.SMTPException("connection refused"),
    )
    # Must not propagate
    decision = machine.run_cycle(_wake())

    assert decision.alarm is True
    alarm_on_spy.assert_called_once()
    notify_spy.assert_called_once()


def test_snapshot_is_rgb_for_vision(monkeypatch):
    """The snapshot passed to vision.verdict must be BGR→RGB-flipped (channel-order contract)."""
    # Build frames where last frame has detectable channel values
    frames = np.zeros((4, 1, 1, 3), dtype=np.uint8)
    frames[-1, 0, 0] = [1, 2, 3]  # BGR [B=1, G=2, R=3]

    monkeypatch.setattr("agent.camera.capture", lambda **kw: frames)
    monkeypatch.setattr("agent.prefilter.run", lambda f: _pf_escalate())
    monkeypatch.setattr("agent.notifier.save_clip", lambda f: _clip_path())
    monkeypatch.setattr("agent.actuators.alarm_on", MagicMock())
    monkeypatch.setattr("agent.notifier.notify", MagicMock())

    captured_snapshot = []

    def spy_verdict(snapshot, **kw):
        captured_snapshot.append(snapshot.copy())
        return VisionVerdict(is_intrusion=False, confidence=0.1, reason="ok", source="gemini")

    monkeypatch.setattr("agent.vision.verdict", spy_verdict)

    machine.run_cycle(_wake())

    assert len(captured_snapshot) == 1
    snap = captured_snapshot[0]
    expected = frames[-1][..., ::-1]  # BGR → RGB
    np.testing.assert_array_equal(snap, expected)


def test_event_log_written(monkeypatch, tmp_path):
    log_path = tmp_path / "event.log"
    monkeypatch.setattr(config, "EVENT_LOG", log_path)

    v = VisionVerdict(is_intrusion=True, confidence=0.95, reason="intruder", source="gemini")
    _patch_leaves(monkeypatch, pf=_pf_escalate(), verdict=v)
    monkeypatch.setattr("agent.actuators.alarm_on", MagicMock())
    monkeypatch.setattr("agent.notifier.notify", MagicMock())

    machine.run_cycle(_wake())

    lines = [l for l in log_path.read_text().splitlines() if l.strip()]
    assert len(lines) >= 1
    record = json.loads(lines[-1])
    assert record["alarm"] is True
