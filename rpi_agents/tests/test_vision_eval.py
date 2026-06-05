"""Offline eval: assert full prefilter-gate + vision-verdict decision metrics.

Person detection is disabled so the gate is deterministic (motion-only).
Vision calls are replayed from committed JSON fixtures — no live API spend.

Acceptance thresholds (from architecture § Evaluation Framework):
  - recall               >= config.RECALL_MIN                   (0.98)
  - false_alarm_rate     <= config.FALSE_ALARM_MAX               (0.20)
  - vision_calls / false <= config.VISION_CALLS_MAX_PER_NONINTRUSION  (0.30)

Eval split: 2 intrusion (moving), 2 false (moving), 6 false (static).
  vision_calls_per_nonintrusion = 2/8 = 0.25 (within budget)
  FAR = 0/8 = 0.0, recall = 1.0
"""

from pathlib import Path

import numpy as np
import pytest

from agent import config, prefilter, vision
from tests.eval_harness import Sample, score

# Fixed dimensions matching test_prefilter_eval.py (fast CI)
_H, _W, _N = 32, 32, 4

# Replay map: sample_id → recorded Gemini JSON string (committed fixtures)
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vision_replay"

_REPLAY: dict[str, str] = {
    sid: (_FIXTURE_DIR / f"{sid}.json").read_text()
    for sid in ("i_eval_0", "i_eval_1", "mf_eval_0", "mf_eval_1")
}


# Synthetic clip generators (no real footage in repo)


def _moving_clip() -> np.ndarray:
    """Bright blob shifts non-overlapping 8 px per frame — clearly above threshold."""
    frames = []
    for i in range(_N):
        f = np.zeros((_H, _W, 3), dtype=np.uint8)
        col = i * 8
        f[8:24, col : col + 8, :] = 255
        frames.append(f)
    return np.stack(frames)


def _static_clip() -> np.ndarray:
    """Flat grey clip — zero inter-frame diff, no motion."""
    return np.full((_N, _H, _W, 3), 128, dtype=np.uint8)


def _make_eval_samples(tmp_path: Path) -> list[Sample]:
    """Write synthetic .npy clips and return a labeled Sample list (eval split only).

    2 intrusion (moving) + 2 false (moving) + 6 false (static).
    vision_calls_per_nonintrusion = 2/8 = 0.25 (within 0.30 budget).
    """
    samples: list[Sample] = []

    for idx in range(2):
        sid = f"i_eval_{idx}"
        p = tmp_path / f"{sid}.npy"
        np.save(str(p), _moving_clip())
        samples.append(Sample(id=sid, path=str(p), label="intrusion", split="eval"))

    for idx in range(2):
        sid = f"mf_eval_{idx}"
        p = tmp_path / f"{sid}.npy"
        np.save(str(p), _moving_clip())
        samples.append(Sample(id=sid, path=str(p), label="false", split="eval"))

    for idx in range(6):
        sid = f"sf_eval_{idx}"
        p = tmp_path / f"{sid}.npy"
        np.save(str(p), _static_clip())
        samples.append(Sample(id=sid, path=str(p), label="false", split="eval"))

    return samples


# Autouse fixture: disable person detection + reset net cache


@pytest.fixture(autouse=True)
def disable_person_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force motion-only gate — deterministic, no model weights needed in CI."""
    monkeypatch.setattr(config, "PREFILTER_PERSON_ENABLED", False)
    monkeypatch.setattr(prefilter, "_net", None)
    monkeypatch.setattr(prefilter, "_net_loaded", False)


# classify_fn: prefilter gate → optional vision replay → (alarm, used_vision)


def _classify(sample: Sample) -> tuple[bool, bool]:
    frames = np.load(sample.path)
    pf = prefilter.run(frames)
    if not pf.escalate:
        return (False, False)
    snapshot = frames[-1]
    replay_json = _REPLAY[sample.id]
    v = vision.verdict(snapshot, generate=lambda _s, j=replay_json: j)
    return (v.is_intrusion, True)


# Gate metric tests


def test_eval_recall_meets_threshold(tmp_path: Path) -> None:
    samples = _make_eval_samples(tmp_path)
    report = score(samples, _classify)
    assert report.recall >= config.RECALL_MIN, (
        f"recall={report.recall:.3f} < threshold {config.RECALL_MIN}"
    )


def test_eval_false_alarm_within_budget(tmp_path: Path) -> None:
    samples = _make_eval_samples(tmp_path)
    report = score(samples, _classify)
    assert report.false_alarm_rate <= config.FALSE_ALARM_MAX, (
        f"false_alarm_rate={report.false_alarm_rate:.3f} > budget {config.FALSE_ALARM_MAX}"
    )


def test_eval_vision_cost_within_budget(tmp_path: Path) -> None:
    samples = _make_eval_samples(tmp_path)
    report = score(samples, _classify)
    assert report.vision_calls_per_nonintrusion <= config.VISION_CALLS_MAX_PER_NONINTRUSION, (
        f"vision_calls_per_nonintrusion={report.vision_calls_per_nonintrusion:.3f} "
        f"> budget {config.VISION_CALLS_MAX_PER_NONINTRUSION}"
    )


def test_eval_full_report_passes(tmp_path: Path) -> None:
    """Combined gate: all three thresholds must be met simultaneously."""
    samples = _make_eval_samples(tmp_path)
    report = score(samples, _classify)
    assert report.passed, (
        f"EvalReport failed: recall={report.recall:.3f}, "
        f"far={report.false_alarm_rate:.3f}, "
        f"vision_per_nonintrusion={report.vision_calls_per_nonintrusion:.3f}"
    )


def test_failsafe_counts_as_alarm(tmp_path: Path) -> None:
    """A replay that raises must trigger failsafe ALARM (source='failsafe')."""
    p = tmp_path / "intrusion_failsafe.npy"
    np.save(str(p), _moving_clip())

    frames = np.load(str(p))
    pf = prefilter.run(frames)
    assert pf.escalate, "Test precondition: moving clip must escalate"

    snapshot = frames[-1]
    def _raise(_s: np.ndarray) -> str:
        raise TimeoutError("simulated")

    v = vision.verdict(snapshot, generate=_raise)
    assert v.is_intrusion is True
    assert v.source == "failsafe"
