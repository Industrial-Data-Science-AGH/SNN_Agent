"""Score a Fourier front end with the project's own deployment metric.

The protocol, which is the whole point of this file:

1. Fit a per frame classifier on **train** only.
2. Choose the detection threshold **and** the k-of-w rule on **val**, by the
   same ``recall at a fixed FA/h budget`` the SNN is selected by.
3. Report that frozen pair on **test**, once.

Step 3 calls ``stream_report`` with a single rule on purpose. Handed the whole
grid, it picks the best one for the data it is given; letting it do that on test
would be choosing the operating point on the test set, which is the mistake the
repository audit found in three places already.

Frame labels are the clip's label broadcast to its frames. That is a handicap,
and it is deliberate: ``snn_hw_pipeline.py:319`` does exactly the same for the
SNN. Giving the Fourier side true frame boundaries from the VOICe annotations
while the SNN trains on clip labels would make the comparison meaningless in the
other direction.

    python -m comparison.evaluate --variant mcu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from comparison.features import FS_HZ, HOP
from snn_pipeline.stream_eval import DEFAULT_RULES, stream_report

DT_S = HOP / FS_HZ
# 1 and 6 FA/h are the project's declared budgets (stream_eval.DEFAULT_BUDGETS).
# The rest are there because both sides may score near zero at those two, and a
# pair of zeros compares nothing. 120 FA/h is the break-even from the energy
# note (DOK 1): above it, waking the reactor costs more than staying awake, so
# it is the point past which the whole architecture stops paying for itself.
BUDGETS = (1.0, 6.0, 30.0, 120.0, 600.0)
# The useful region is the far tail: at a 1 FA/h budget the detector may fire
# on a handful of background frames in the whole test set, so a grid that stops
# at 0.95 leaves the baseline undertuned and the comparison unfair. The tail is
# therefore refined geometrically towards 1.
THRESHOLDS = tuple(
    sorted(
        {round(float(x), 6) for x in np.arange(0.05, 1.0, 0.05)}
        | {round(float(1.0 - 10.0**-e), 6) for e in np.arange(1.0, 5.01, 0.25)}
    )
)


@dataclass
class Split:
    features: np.ndarray
    lengths: np.ndarray
    label: np.ndarray
    kind: np.ndarray
    group: np.ndarray

    @property
    def offsets(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.lengths)])

    def frame_labels(self) -> np.ndarray:
        return np.repeat(self.label, self.lengths)

    def per_clip(self, values: np.ndarray) -> list[np.ndarray]:
        bounds = self.offsets
        return [values[bounds[i] : bounds[i + 1]] for i in range(len(self.lengths))]


def load(cache: Path, variant: str, split: str) -> Split:
    data = np.load(cache / f"{variant}-{split}.npz", allow_pickle=True)
    return Split(
        features=data["features"],
        lengths=data["lengths"],
        label=data["label"].astype(int),
        kind=data["kind"].astype(object),
        group=data["group"].astype(object),
    )


def fit(train: Split, max_frames: int, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    x, y = train.features, train.frame_labels()
    if len(x) > max_frames:
        # Subsampling frames, never clips: dropping whole clips would change
        # which groups the model ever sees.
        pick = np.random.default_rng(seed).choice(len(x), size=max_frames, replace=False)
        x, y = x[pick], y[pick]
    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.1, max_leaf_nodes=31, early_stopping=True, random_state=seed
    )
    model.fit(x, y)
    return model


def trains_at(scores: Split, probability: np.ndarray, threshold: float) -> list[np.ndarray]:
    return scores.per_clip(probability >= threshold)


def choose_operating_point(val: Split, probability: np.ndarray, budget: float) -> dict:
    """Threshold and rule, both picked on val, never on test."""
    best = {"recall": -1.0, "threshold": None, "rule": None, "feasible": False}
    for threshold in THRESHOLDS:
        report = stream_report(
            trains_at(val, probability, threshold), val.label, val.kind, val.group,
            n_frames=val.lengths, dt=DT_S, rules=DEFAULT_RULES, budgets=(budget,), n_boot=0,
        )  # fmt: skip
        point = report[budget]
        if point.feasible and point.recall > best["recall"]:
            best = {
                "recall": float(point.recall), "threshold": float(threshold),
                "rule": list(point.rule), "feasible": True,
            }  # fmt: skip
    return best


def report_on_test(test: Split, probability: np.ndarray, chosen: dict, budget: float) -> dict:
    if not chosen["feasible"]:
        return {"feasible": False, "note": f"no rule stayed within {budget} FA/h on val"}
    rule = tuple(chosen["rule"])
    report = stream_report(
        trains_at(test, probability, chosen["threshold"]), test.label, test.kind, test.group,
        n_frames=test.lengths, dt=DT_S, rules=(rule,), budgets=(budget,), n_boot=500,
    )  # fmt: skip
    point = report[budget]
    if not point.feasible:
        # `_recall_at_budget` returns a default point when no rule stayed within
        # the budget, and its zeros are placeholders, not measurements. Printing
        # them as "recall 0.0 at 0.0 FA/h" reads like a clean miss when it
        # actually means the point chosen on val did not hold on test.
        return {
            "feasible": False,
            "threshold": chosen["threshold"],
            "rule": list(rule),
            "note": f"the pair chosen on val exceeded {budget} FA/h on test; no recall is reported",
        }
    return {
        "feasible": True,
        "threshold": chosen["threshold"],
        "rule": list(rule),
        "recall": round(point.recall, 4),
        "recall_ci": [round(v, 4) for v in (point.recall_ci or (float("nan"),) * 2)],
        "fa_h_total": round(point.fa_h_total, 3),
        "fa_h_by_kind": {k: round(v, 3) for k, v in sorted(point.fa_h_by_kind.items())},
        "within_budget": point.feasible,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--cache", type=Path, default=Path("comparison/cache"))
    parser.add_argument("--out", type=Path, default=Path("comparison/results"))
    parser.add_argument("--max-train-frames", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    train = load(args.cache, args.variant, "train")
    val = load(args.cache, args.variant, "val")
    test = load(args.cache, args.variant, "test")
    print(
        f"{args.variant}: train {len(train.lengths)} clips / {train.lengths.sum()} frames, "
        f"val {len(val.lengths)}, test {len(test.lengths)}, dt={DT_S * 1e6:.0f} us",
        flush=True,
    )

    scores = args.cache / f"{args.variant}-scores-s{args.seed}.npz"
    if scores.exists():
        cached = np.load(scores)
        p_val, p_test = cached["val"], cached["test"]
        print(f"  reusing {scores}", flush=True)
    else:
        model = fit(train, args.max_train_frames, args.seed)
        p_val = model.predict_proba(val.features)[:, 1]
        p_test = model.predict_proba(test.features)[:, 1]
        np.savez(scores, val=p_val, test=p_test)

    result = {
        "variant": args.variant,
        "dataset": "dataset/versions/v2.0.0",
        "dt_us": round(DT_S * 1e6, 1),
        "seed": args.seed,
        "frame_label": "clip label broadcast to frames, as in snn_hw_pipeline.py:319",
        "test_background_hours": round(float(test.lengths[test.label == 0].sum()) * DT_S / 3600, 3),
        "budgets": {},
    }
    for budget in BUDGETS:
        chosen = choose_operating_point(val, p_val, budget)
        result["budgets"][str(budget)] = {"chosen_on_val": chosen, "test": report_on_test(test, p_test, chosen, budget)}
        print(f"  budget {budget} FA/h -> {result['budgets'][str(budget)]['test']}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.variant}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
