"""Unit tests for agent.power: cooldown sleep and power-mode-aware halt."""

from agent import config, power


def test_cooldown_zero_returns_immediately(monkeypatch):
    calls = []
    monkeypatch.setattr("time.sleep", lambda s: calls.append(s))
    power.cooldown(0)
    assert calls == [0]


def test_resleep_warm_does_not_call_halt(monkeypatch):
    monkeypatch.setattr(config, "POWER_MODE", "warm")
    spy = []
    monkeypatch.setattr("subprocess.run", lambda cmd, **_: spy.append(cmd))
    power.resleep()
    assert spy == []


def test_resleep_halt_calls_sudo_halt(monkeypatch):
    monkeypatch.setattr(config, "POWER_MODE", "halt")
    spy = []

    class FakeProc:
        returncode = 0

    monkeypatch.setattr("subprocess.run", lambda cmd, **_: (spy.append(cmd), FakeProc())[1])
    power.resleep()
    assert spy == [["sudo", "halt"]]


def test_resleep_halt_raises_on_nonzero(monkeypatch):
    import pytest

    monkeypatch.setattr(config, "POWER_MODE", "halt")

    class FakeProc:
        returncode = 1

    monkeypatch.setattr("subprocess.run", lambda _cmd, **_: FakeProc())
    with pytest.raises(RuntimeError, match="returncode 1"):
        power.resleep()
