"""Mocked end-to-end tests for agent.main (P4)."""

import sys
from unittest.mock import MagicMock, call

import pytest

from agent import config, main
from agent.types import Decision, WakeContext


def _wake() -> WakeContext:
    return WakeContext(woken_by_trigger=True, ts_monotonic=0.0, ts_wall=0.0)


def _patch_production(monkeypatch, *, run_cycle_result=None, run_cycle_raises=None):
    """Patch all production-path collaborators; return spies."""
    monkeypatch.setattr(
        "agent.trigger.read_wake_context",
        lambda: _wake(),
    )

    if run_cycle_raises is not None:
        monkeypatch.setattr(
            "agent.machine.run_cycle",
            MagicMock(side_effect=run_cycle_raises),
        )
    else:
        monkeypatch.setattr(
            "agent.machine.run_cycle",
            MagicMock(return_value=run_cycle_result),
        )

    resleep_spy = MagicMock()
    monkeypatch.setattr("agent.power.resleep", resleep_spy)

    cooldown_spy = MagicMock()
    monkeypatch.setattr("agent.power.cooldown", cooldown_spy)

    close_spy = MagicMock()
    monkeypatch.setattr("agent.actuators.close", close_spy)

    return resleep_spy, cooldown_spy, close_spy


def test_e2e_alarm_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main"])
    resleep_spy, cooldown_spy, close_spy = _patch_production(
        monkeypatch, run_cycle_result=Decision(alarm=True, reason="person at window")
    )

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0
    resleep_spy.assert_called_once()
    close_spy.assert_called_once()
    # cooldown called with ALARM_HOLD_S AND COOLDOWN_S
    assert call(config.ALARM_HOLD_S) in cooldown_spy.call_args_list
    assert call(config.COOLDOWN_S) in cooldown_spy.call_args_list


def test_e2e_false_path(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main"])
    resleep_spy, cooldown_spy, close_spy = _patch_production(
        monkeypatch, run_cycle_result=Decision(alarm=False, reason="static")
    )

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0
    resleep_spy.assert_called_once()
    close_spy.assert_called_once()
    # ALARM_HOLD_S must NOT appear; COOLDOWN_S must appear
    assert call(config.ALARM_HOLD_S) not in cooldown_spy.call_args_list
    assert call(config.COOLDOWN_S) in cooldown_spy.call_args_list


def test_always_resleeps_on_cycle_error(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main"])
    resleep_spy, cooldown_spy, close_spy = _patch_production(
        monkeypatch, run_cycle_raises=RuntimeError("camera failed")
    )

    with pytest.raises(RuntimeError, match="camera failed"):
        main.main()

    # Pi must still shut down even on mid-cycle crash
    close_spy.assert_called_once()
    resleep_spy.assert_called_once()


def test_smoke_test_still_passes(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["main", "--test"])

    with pytest.raises(SystemExit) as exc:
        main.main()

    assert exc.value.code == 0
