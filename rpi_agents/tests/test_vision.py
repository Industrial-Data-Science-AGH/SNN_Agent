"""Unit tests for agent.vision (P3)."""

import numpy as np
import pytest

from agent import vision

_SNAPSHOT = np.zeros((32, 32, 3), dtype=np.uint8)


def test_parse_good_json() -> None:
    raw = (
        '{"window_broken": true, "is_intrusion": true, "confidence": 0.9, '
        '"reason": "person at window"}'
    )
    v = vision._parse_verdict(raw)
    assert v.is_intrusion is True
    assert v.window_broken is True
    assert v.confidence == pytest.approx(0.9)
    assert v.reason == "person at window"
    assert v.source == "gemini"


def test_parse_rejects_missing_field() -> None:
    with pytest.raises((ValueError, KeyError)):
        vision._parse_verdict('{"window_broken": true, "is_intrusion": true, "confidence": 0.9}')


def test_parse_rejects_missing_window_broken() -> None:
    """window_broken (2026-07-15, ADR-0017) is required from Gemini, same as
    the other three fields — no silent default in the real parse path."""
    with pytest.raises(ValueError, match="window_broken"):
        vision._parse_verdict('{"is_intrusion": true, "confidence": 0.9, "reason": "x"}')


def test_parse_strips_code_fence() -> None:
    raw = (
        '```json\n{"window_broken": false, "is_intrusion": false, '
        '"confidence": 0.1, "reason": "cat"}\n```'
    )
    v = vision._parse_verdict(raw)
    assert v.is_intrusion is False
    assert v.window_broken is False
    assert v.source == "gemini"


def test_parse_clamps_confidence() -> None:
    raw = '{"window_broken": true, "is_intrusion": true, "confidence": 1.5, "reason": "overflow"}'
    v = vision._parse_verdict(raw)
    assert v.confidence == pytest.approx(1.0)


def test_verdict_happy_path_source_gemini() -> None:
    v = vision.verdict(
        _SNAPSHOT,
        generate=lambda _s: (
            '{"window_broken": true, "is_intrusion": true, "confidence": 0.9, '
            '"reason": "person at window"}'
        ),
    )
    assert v.is_intrusion is True
    assert v.window_broken is True
    assert v.source == "gemini"


def test_verdict_false_pass_through() -> None:
    v = vision.verdict(
        _SNAPSHOT,
        generate=lambda _s: (
            '{"window_broken": false, "is_intrusion": false, "confidence": 0.1, "reason": "cat"}'
        ),
    )
    assert v.is_intrusion is False
    assert v.window_broken is False
    assert v.source == "gemini"


def test_verdict_window_broken_independent_of_is_intrusion() -> None:
    """The two judgments are independent -- a broken window with no
    confirmed intruder must not force is_intrusion, and vice versa."""
    v = vision.verdict(
        _SNAPSHOT,
        generate=lambda _s: (
            '{"window_broken": true, "is_intrusion": false, "confidence": 0.4, '
            '"reason": "broken pane, no one visible"}'
        ),
    )
    assert v.window_broken is True
    assert v.is_intrusion is False


def test_verdict_failsafe_on_error() -> None:
    def _raise(_s: np.ndarray) -> str:
        raise TimeoutError("simulated timeout")

    v = vision.verdict(_SNAPSHOT, generate=_raise)
    assert v.is_intrusion is True
    assert v.window_broken is True  # fail-open, same conservative default as is_intrusion
    assert v.source == "failsafe"
    assert "TimeoutError" in v.reason


def test_verdict_failsafe_on_bad_json() -> None:
    v = vision.verdict(_SNAPSHOT, generate=lambda _s: "not json")
    assert v.is_intrusion is True
    assert v.source == "failsafe"


def test_verdict_failsafe_confidence_is_one() -> None:
    """Failsafe verdicts always carry confidence=1.0 to pin recall."""
    v = vision.verdict(_SNAPSHOT, generate=lambda _s: "broken")
    assert v.confidence == pytest.approx(1.0)


def test_import_has_no_google_dep() -> None:
    """agent.vision must import cleanly without google-genai installed."""
    import agent.vision as av  # noqa: F401 — asserts no ImportError at load time

    assert av is not None
