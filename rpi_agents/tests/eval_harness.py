"""Offline decision-quality eval harness.

Scores a classify_fn against a labeled manifest (CSV) without opening any
image files — the classify_fn owns all I/O.  Run on the Mac with no hardware.

Usage:
    from tests.eval_harness import load_manifest, score, RECALL_MIN

Thresholds are inherited from architecture § Evaluation Framework and mirrored
in agent/config.py.  P0 gate = harness is correct + package import-safe;
no real classifier is scored yet.
"""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal


# ACCEPTANCE THRESHOLDS (verbatim from architecture § Evaluation Framework)


RECALL_MIN: float = 0.98
"""Minimum recall (TP / (TP+FN)) required for the eval set to pass."""

FALSE_ALARM_MAX: float = 0.20
"""Maximum false-alarm rate (FP / (FP+TN)) allowed to pass."""

VISION_CALLS_MAX_PER_NONINTRUSION: float = 0.30
"""Maximum fraction of non-intrusion events that may trigger a vision call."""


# DATA CONTRACTS


@dataclass(frozen=True)
class Sample:
    """One labeled example from the eval manifest."""

    id: str
    path: str
    label: Literal["intrusion", "false"]
    split: Literal["tune", "eval"]


@dataclass(frozen=True)
class EvalReport:
    """Aggregated eval metrics for a single scorer run."""

    n: int
    recall: float
    false_alarm_rate: float
    vision_calls_per_nonintrusion: float
    passed: bool


# MANIFEST LOADER


def load_manifest(csv_path: str | Path) -> list[Sample]:
    """Parse a labels CSV into a list of Sample objects.

    Expected columns: id, path, label, split.

    Args:
        csv_path: Path to the manifest CSV file.

    Returns:
        List of Sample dataclasses, one per row (header skipped).
    """
    samples: list[Sample] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            samples.append(Sample(
                id=row["id"],
                path=row["path"],
                label=row["label"],      # type: ignore[arg-type]
                split=row["split"],      # type: ignore[arg-type]
            ))
    return samples


# SCORER


def score(
    samples: list[Sample],
    classify_fn: Callable[[Sample], tuple[bool, bool]],
) -> EvalReport:
    """Score classify_fn against labeled samples.

    Args:
        samples: List of labeled samples (from load_manifest or synthetic).
        classify_fn: Callable(sample) -> (alarm: bool, used_vision: bool).
            The function owns all image I/O; the harness never opens files.

    Returns:
        EvalReport with recall, false_alarm_rate, vision_calls_per_nonintrusion,
        and a passed flag (True only when all three thresholds are met and
        both classes are non-empty).

    Notes:
        - recall        = TP / (TP+FN)  over intrusion samples
        - false_alarm   = FP / (FP+TN)  over false samples
        - vision_calls  = Σ(used_vision for false samples) / n_false
        - If a class is empty, the affected metric is 0.0 and passed=False.
    """
    tp = fp = tn = fn = 0
    vision_on_false = 0
    n_false = 0
    n_intrusion = 0

    for sample in samples:
        alarm, used_vision = classify_fn(sample)

        if sample.label == "intrusion":
            n_intrusion += 1
            if alarm:
                tp += 1
            else:
                fn += 1
        else:
            n_false += 1
            if alarm:
                fp += 1
            else:
                tn += 1
            if used_vision:
                vision_on_false += 1

    # Guard div-by-zero: empty class → metric defaults that force passed=False.
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    vision_calls_per_nonintrusion = vision_on_false / n_false if n_false > 0 else 0.0

    both_classes_present = n_intrusion > 0 and n_false > 0
    passed = (
        both_classes_present
        and recall >= RECALL_MIN
        and false_alarm_rate <= FALSE_ALARM_MAX
        and vision_calls_per_nonintrusion <= VISION_CALLS_MAX_PER_NONINTRUSION
    )

    return EvalReport(
        n=len(samples),
        recall=recall,
        false_alarm_rate=false_alarm_rate,
        vision_calls_per_nonintrusion=vision_calls_per_nonintrusion,
        passed=passed,
    )
