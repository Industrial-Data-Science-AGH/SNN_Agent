"""Unit tests for agent.prefilter: motion, person detection, graceful degrade, determinism.

Person net is monkeypatched — no weights required in CI.
"""

import numpy as np
import pytest

from agent import config, prefilter
from agent.types import PrefilterResult

_H, _W, _N = 32, 32, 4  # tiny synthetic frames for fast CI


# Synthetic clip builders


def _static_frames() -> np.ndarray:
    """N identical frames — zero inter-frame diff, no motion."""
    return np.full((_N, _H, _W, 3), 128, dtype=np.uint8)


def _moving_frames() -> np.ndarray:
    """N frames with a non-overlapping bright blob — clearly above threshold."""
    frames = []
    for i in range(_N):
        f = np.zeros((_H, _W, 3), dtype=np.uint8)
        col = i * 8  # non-overlapping 8-pixel shifts across 32-px width
        f[8:24, col : col + 8, :] = 255
        frames.append(f)
    return np.stack(frames)


# Autouse fixture: reset module-level net cache + last_detection before each test


@pytest.fixture(autouse=True)
def reset_net_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prefilter, "_net", None)
    monkeypatch.setattr(prefilter, "_net_loaded", False)
    monkeypatch.setattr(prefilter, "last_detection", {})


# Motion detection


def test_static_frames_no_motion() -> None:
    result = prefilter.run(_static_frames())
    assert isinstance(result, PrefilterResult)
    assert result.motion is False
    assert result.score == pytest.approx(0.0)


def test_moving_frames_detect_motion() -> None:
    result = prefilter.run(_moving_frames())
    assert result.motion is True
    assert result.score > config.PREFILTER_MOTION_THRESHOLD


def test_single_frame_no_motion() -> None:
    frame = np.full((1, _H, _W, 3), 64, dtype=np.uint8)
    result = prefilter.run(frame)
    assert result.motion is False


def test_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same input twice must produce identical PrefilterResult."""
    monkeypatch.setattr(config, "PREFILTER_PERSON_ENABLED", False)
    frames = _moving_frames()
    r1 = prefilter.run(frames)
    # Reset cache to force _load_net() to re-run the same code path
    monkeypatch.setattr(prefilter, "_net_loaded", False)
    r2 = prefilter.run(frames)
    assert r1 == r2


# Person detection (net mocked — no weights)


class _FakeNet:
    """Simulates a MobileNet-SSD net that reports one 'person' detection."""

    def __init__(self, conf: float = 0.9) -> None:
        self._conf = conf

    def setInput(self, _blob: np.ndarray) -> None:
        pass

    def forward(self) -> np.ndarray:
        # Shape (1, 1, N_det, 7); class 15 = person
        det = np.zeros((1, 1, 1, 7), dtype=np.float32)
        det[0, 0, 0] = [0, 15, self._conf, 0.1, 0.1, 0.5, 0.5]
        return det


def test_person_detection_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Injecting a mocked net that returns a person detection → person=True, escalate=True."""
    monkeypatch.setattr(prefilter, "_net", _FakeNet(conf=0.9))
    monkeypatch.setattr(prefilter, "_net_loaded", True)

    result = prefilter.run(_static_frames())
    assert result.person is True
    assert result.escalate is True
    assert prefilter.last_detection.get("confidence") == pytest.approx(0.9, abs=1e-5)


def test_low_confidence_person_not_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Detection below PREFILTER_PERSON_CONF must not set person=True."""
    low_conf = config.PREFILTER_PERSON_CONF - 0.1
    monkeypatch.setattr(prefilter, "_net", _FakeNet(conf=low_conf))
    monkeypatch.setattr(prefilter, "_net_loaded", True)

    result = prefilter.run(_static_frames())
    assert result.person is False


def test_model_missing_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pytest.TempPathFactory
) -> None:
    """Missing model files → person=False, no exception; motion still governs escalate."""
    monkeypatch.setattr(config, "PREFILTER_PERSON_ENABLED", True)
    monkeypatch.setattr(config, "PERSON_MODEL_DIR", tmp_path)  # empty dir

    result = prefilter.run(_moving_frames())
    assert result.person is False
    assert result.motion is True
    assert result.escalate is True  # motion-driven escalation preserved


def test_person_detection_disabled_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PREFILTER_PERSON_ENABLED=False → person=False, no raise."""
    monkeypatch.setattr(config, "PREFILTER_PERSON_ENABLED", False)

    result = prefilter.run(_static_frames())
    assert result.person is False
    assert result.escalate is False


# last_detection bookkeeping


def test_escalate_true_populates_last_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Moving frames → escalate=True → last_detection updated with frame_idx."""
    monkeypatch.setattr(config, "PREFILTER_PERSON_ENABLED", False)
    prefilter.run(_moving_frames())
    assert "frame_idx" in prefilter.last_detection


def test_escalate_false_does_not_update_last_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Static frames, no person → escalate=False → last_detection stays empty."""
    monkeypatch.setattr(config, "PREFILTER_PERSON_ENABLED", False)
    prefilter.run(_static_frames())
    assert prefilter.last_detection == {}


def test_person_escalation_populates_bbox(monkeypatch: pytest.MonkeyPatch) -> None:
    """Person detection → last_detection includes bbox and confidence."""
    monkeypatch.setattr(prefilter, "_net", _FakeNet(conf=0.9))
    monkeypatch.setattr(prefilter, "_net_loaded", True)

    prefilter.run(_static_frames())
    ld = prefilter.last_detection
    assert "bbox" in ld
    assert ld["bbox"] is not None
    assert "confidence" in ld
