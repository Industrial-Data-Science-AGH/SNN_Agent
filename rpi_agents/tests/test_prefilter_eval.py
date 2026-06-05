"""Offline eval: assert prefilter gate metrics on synthetic labeled clips.

Person detection is disabled so the gate is exercised on motion alone
(deterministic, no weights required in CI).

Acceptance thresholds (from architecture § Evaluation Framework):
  - recall == 1.0   (permissive gate must never drop a moving intrusion)
  - vision_calls_per_nonintrusion ≤ config.VISION_CALLS_MAX_PER_NONINTRUSION
    (static false-alarm scenes must not escalate to vision)
"""

import numpy as np
import pytest

from agent import config, prefilter
from tests.eval_harness import Sample, score

# Fixed dimensions for fast CI (clips are fully deterministic — no RNG needed)
_H, _W, _N = 32, 32, 4  # (n_frames, height, width)


# Synthetic clip generators (committed code, no real footage in repo)


def _static_clip() -> np.ndarray:
    """Flat grey clip — zero inter-frame diff, no motion."""
    return np.full((_N, _H, _W, 3), 128, dtype=np.uint8)


def _moving_clip() -> np.ndarray:
    """Bright blob shifts non-overlapping 8 px per frame — clearly above threshold."""
    frames = []
    for i in range(_N):
        f = np.zeros((_H, _W, 3), dtype=np.uint8)
        col = i * 8
        f[8:24, col : col + 8, :] = 255
        frames.append(f)
    return np.stack(frames)


def _make_samples(tmp_path: object) -> list[Sample]:
    """Write synthetic .npy clips to tmp_path and return a labeled Sample list.

    4 intrusion (moving) + 4 false (static), split evenly across tune / eval.
    """
    from pathlib import Path

    base = Path(str(tmp_path))
    samples: list[Sample] = []

    for idx in range(4):
        p = base / f"intrusion_{idx}.npy"
        np.save(str(p), _moving_clip())
        split = "tune" if idx < 2 else "eval"
        samples.append(Sample(id=f"i_{idx}", path=str(p), label="intrusion", split=split))

    for idx in range(4):
        p = base / f"false_{idx}.npy"
        np.save(str(p), _static_clip())
        split = "tune" if idx < 2 else "eval"
        samples.append(Sample(id=f"f_{idx}", path=str(p), label="false", split=split))

    return samples


# Autouse fixture: disable person detection + reset net cache


@pytest.fixture(autouse=True)
def disable_person_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force motion-only gate — deterministic, no weights needed in CI."""
    monkeypatch.setattr(config, "PREFILTER_PERSON_ENABLED", False)
    monkeypatch.setattr(prefilter, "_net", None)
    monkeypatch.setattr(prefilter, "_net_loaded", False)


# classify_fn: load clip from disk, run prefilter, return (alarm, used_vision)


def _classify(sample: Sample) -> tuple[bool, bool]:
    frames = np.load(sample.path)
    result = prefilter.run(frames)
    return result.escalate, result.escalate


# Gate metric tests


def test_prefilter_gate_recall(tmp_path: pytest.TempPathFactory) -> None:
    """Permissive gate must escalate every moving intrusion clip (recall == 1.0)."""
    samples = _make_samples(tmp_path)
    report = score(samples, _classify)
    assert report.recall == 1.0, (
        f"Gate dropped an intrusion: recall={report.recall:.3f} (expected 1.0)"
    )


def test_prefilter_gate_vision_cost(tmp_path: pytest.TempPathFactory) -> None:
    """Static false-alarm clips must not escalate beyond the vision-cost budget."""
    samples = _make_samples(tmp_path)
    report = score(samples, _classify)
    assert report.vision_calls_per_nonintrusion <= config.VISION_CALLS_MAX_PER_NONINTRUSION, (
        f"vision_calls_per_nonintrusion={report.vision_calls_per_nonintrusion:.3f} "
        f"> budget {config.VISION_CALLS_MAX_PER_NONINTRUSION}"
    )


def test_prefilter_gate_full_report_passes(tmp_path: pytest.TempPathFactory) -> None:
    """Combined: recall, false-alarm-rate, and vision-cost all within thresholds."""
    samples = _make_samples(tmp_path)
    report = score(samples, _classify)
    assert report.passed, (
        f"EvalReport failed: recall={report.recall:.3f}, "
        f"far={report.false_alarm_rate:.3f}, "
        f"vision_per_nonintrusion={report.vision_calls_per_nonintrusion:.3f}"
    )
