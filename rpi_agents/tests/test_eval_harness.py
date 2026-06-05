"""Unit tests for the eval harness scorer and manifest loader."""

from pathlib import Path

import pytest

from tests.eval_harness import (
    RECALL_MIN,
    FALSE_ALARM_MAX,
    VISION_CALLS_MAX_PER_NONINTRUSION,
    Sample,
    load_manifest,
    score,
)

# Helpers

_FIXTURES_CSV = Path(__file__).parent / "fixtures" / "labels.csv"


def _make_samples(
    n_intrusion: int,
    n_false: int,
    split: str = "eval",
) -> list[Sample]:
    samples = [
        Sample(id=f"i{i}", path=f"img/i{i}.jpg", label="intrusion", split=split)  # type: ignore[arg-type]
        for i in range(n_intrusion)
    ]
    samples += [
        Sample(id=f"f{i}", path=f"img/f{i}.jpg", label="false", split=split)  # type: ignore[arg-type]
        for i in range(n_false)
    ]
    return samples


# Scorer: perfect classifier


def test_perfect_classifier_passes() -> None:
    samples = _make_samples(5, 5)

    def classify(s: Sample) -> tuple[bool, bool]:
        return s.label == "intrusion", False

    report = score(samples, classify)

    assert report.recall == 1.0
    assert report.false_alarm_rate == 0.0
    assert report.vision_calls_per_nonintrusion == 0.0
    assert report.passed is True
    assert report.n == 10


# Scorer: all false-negative classifier fails


def test_all_false_negative_fails() -> None:
    samples = _make_samples(4, 4)

    def classify(_s: Sample) -> tuple[bool, bool]:
        return False, False  # never alarm

    report = score(samples, classify)

    assert report.recall == 0.0
    assert report.passed is False



# Scorer: vision_calls_per_nonintrusion math


def test_vision_calls_math() -> None:
    # 10 false samples; vision called on 3 of them → 0.3
    samples = _make_samples(5, 10)
    vision_counter = {"n": 0}

    def classify(sample: Sample) -> tuple[bool, bool]:
        if sample.label == "false":
            used = vision_counter["n"] < 3
            vision_counter["n"] += 1 if used else 0
            return False, used
        return True, False

    report = score(samples, classify)

    assert report.vision_calls_per_nonintrusion == pytest.approx(0.3)


# Scorer: threshold boundary — just below recall fails


def test_recall_below_threshold_fails() -> None:
    # 100 intrusion, miss 3 → recall = 0.97 < 0.98
    n = 100
    samples = _make_samples(n, 10)
    missed = {"n": 0}

    def classify(s: Sample) -> tuple[bool, bool]:
        if s.label == "intrusion":
            if missed["n"] < 3:
                missed["n"] += 1
                return False, False
            return True, False
        return False, False

    report = score(samples, classify)

    assert report.recall == pytest.approx(0.97)
    assert report.passed is False


# Scorer: empty intrusion class → passed=False, no exception

def test_empty_intrusion_class_no_exception() -> None:
    samples = _make_samples(0, 5)

    def classify(_s: Sample) -> tuple[bool, bool]:
        return False, False

    report = score(samples, classify)

    assert report.recall == 0.0
    assert report.passed is False



# Scorer: empty false class → passed=False, no exception


def test_empty_false_class_no_exception() -> None:
    samples = _make_samples(5, 0)

    def classify(_s: Sample) -> tuple[bool, bool]:
        return True, False

    report = score(samples, classify)

    assert report.false_alarm_rate == 0.0
    assert report.passed is False



# Manifest loader: parses committed CSV


def test_load_manifest_parses_csv() -> None:
    samples = load_manifest(_FIXTURES_CSV)

    assert len(samples) == 4
    ids = {s.id for s in samples}
    assert ids == {"img_001", "img_002", "img_003", "img_004"}
    labels = {s.label for s in samples}
    assert labels == {"intrusion", "false"}
    splits = {s.split for s in samples}
    assert splits == {"tune", "eval"}


def test_load_manifest_sample_types() -> None:
    samples = load_manifest(_FIXTURES_CSV)
    for s in samples:
        assert isinstance(s, Sample)
        assert s.label in ("intrusion", "false")
        assert s.split in ("tune", "eval")



# Threshold constants match architecture

def test_threshold_constants() -> None:
    assert RECALL_MIN == pytest.approx(0.98)
    assert FALSE_ALARM_MAX == pytest.approx(0.20)
    assert VISION_CALLS_MAX_PER_NONINTRUSION == pytest.approx(0.30)