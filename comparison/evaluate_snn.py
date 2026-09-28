"""Score the SNN D-neuron decoder with the same protocol as comparison/evaluate.py.

The Fourier side (`comparison/evaluate.py`) has a free parameter (probability
threshold) chosen on val and frozen on test. The SNN side has no probability at
all -- the D neuron's output is already a binary spike train -- so its free
parameter is the decoder rule (k spikes in a window of w frames), the same grid
`snn_hw_pipeline.py` and `ga_neuron_search/fitness.py` already select from. This
script applies the identical protocol to that different parameter:

1. Run the checkpoint over every clip in val and test, once, to get D's spike
   train per clip (`stream_eval_torch.d_spike_trains`, unchanged, and generic
   over model class -- it only needs model(x)["so"]).
2. Choose the (k, w) rule **on val**, per FA/h budget: the rule with the
   highest recall among those where every background `kind` stays within
   budget (`snn_pipeline.stream_eval._recall_at_budget`, unchanged -- this is
   the same function GA fitness and checkpoint selection already use).
3. Report that frozen rule **on test**, once, with the CI bootstrap.

This closes the gap `comparison/RESULTS.md` names explicitly: the SNN column
so far was quoted from `models/WYNIKI.md` (PR #49), computed with a hardcoded
k=1 decoder and no val/test split for the operating point. This script
reproduces the number with this harness's protocol instead of citing it.

The checkpoint is a GA-searched topology (`GenomeNet`), not the fixed-shape
`LuiNet` -- `eval_stream.py` only handles the latter, so this script rebuilds
the model from the checkpoint's own `topology` field via `ga_neuron_search`'s
`Genome`/`GenomeNet` instead of importing eval_stream's loader.

    python -m comparison.evaluate_snn --ckpt <path to champion checkpoint>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ARCH_DIR = Path(__file__).resolve().parent.parent / "architecture_14_neurons_patryk_09_07"
_GA_DIR = Path(__file__).resolve().parent.parent / "ga_neuron_search"
for _p in (_ARCH_DIR, _GA_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from genome import Genome  # noqa: E402
from net import GenomeNet  # noqa: E402
from snn_hw_pipeline import DT  # noqa: E402
from snn_pipeline.stream_eval import DEFAULT_RULES, stream_report  # noqa: E402
from snn_pipeline.stream_eval_torch import d_spike_trains, load_clip_spikes, load_files_meta  # noqa: E402

BUDGETS = (1.0, 6.0, 30.0, 120.0, 600.0)  # same grid as comparison/evaluate.py
REFRAC_FRAMES = 500  # 5 s dead zone after an alarm; same constant as eval_stream.py


def load_split(data_root: Path, split: str, ch_in: int):
    clips = load_clip_spikes(str(data_root / split), ch_in)
    labels = np.array([c[1] for c in clips])
    try:
        meta = load_files_meta(str(data_root / split))
    except FileNotFoundError:
        meta = {}
    kinds, groups = [], []
    for _, y, base in clips:
        m = meta.get(base)
        if m is not None:
            kinds.append(m.kind)
            groups.append(m.group_id)
        else:
            kinds.append("positive" if y == 1 else "background")
            groups.append(base)
    return clips, labels, np.array(kinds, dtype=object), np.array(groups, dtype=object)


def choose_operating_point(report, budget: float) -> dict:
    op = report[budget]
    return {
        "recall": float(op.recall),
        "rule": list(op.rule) if op.rule else None,
        "feasible": bool(op.feasible),
    }


def report_on_test(test_report, chosen: dict, budget: float) -> dict:
    if not chosen["feasible"]:
        return {"feasible": False, "note": f"no rule stayed within {budget} FA/h on val"}
    point = test_report[budget]
    if not point.feasible:
        return {
            "feasible": False,
            "rule": chosen["rule"],
            "note": f"the rule chosen on val exceeded {budget} FA/h on test; no recall is reported",
        }
    return {
        "feasible": True,
        "rule": list(point.rule),
        "recall": round(point.recall, 4),
        "recall_ci": [round(v, 4) for v in (point.recall_ci or (float("nan"),) * 2)],
        "fa_h_total": round(point.fa_h_total, 3),
        "fa_h_by_kind": {k: round(v, 3) for k, v in sorted(point.fa_h_by_kind.items())},
        "within_budget": point.feasible,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("architecture_14_neurons_patryk_09_07/spikes_v2"))
    parser.add_argument("--out", type=Path, default=Path("comparison/results"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--boot", type=int, default=500)
    args = parser.parse_args(argv)

    state = torch.load(args.ckpt, map_location="cpu")
    sd = state["model"] if "model" in state else state
    topology = state.get("topology") if isinstance(state, dict) else None
    if topology is None:
        raise SystemExit(f"{args.ckpt} has no 'topology' field -- cannot rebuild GenomeNet")
    genome = Genome.from_dict(topology if "genome" not in topology else topology["genome"])
    model = GenomeNet(genome, quantize=True).to(args.device)
    model.load_state_dict(sd)
    model.eval()
    ch_in = genome.layer_sizes()[0]

    val_clips, val_labels, val_kinds, val_groups = load_split(args.data_root, "val", ch_in)
    test_clips, test_labels, test_kinds, test_groups = load_split(args.data_root, "test", ch_in)
    print(f"snn: val {len(val_clips)} clips, test {len(test_clips)} clips, ch_in={ch_in}, ckpt={args.ckpt}", flush=True)

    val_trains = d_spike_trains(model, val_clips, ch_in, args.device)
    test_trains = d_spike_trains(model, test_clips, ch_in, args.device)
    val_nframes = np.array([len(t) for t in val_trains], dtype=np.int64)
    test_nframes = np.array([len(t) for t in test_trains], dtype=np.int64)

    val_report = stream_report(
        val_trains, val_labels, val_kinds, val_groups, n_frames=val_nframes,
        dt=DT, rules=DEFAULT_RULES, budgets=BUDGETS, refrac=REFRAC_FRAMES, n_boot=0,
    )

    result = {
        "variant": "snn",
        "dataset": "dataset/versions/v2.0.0",
        "dt_us": round(DT * 1e6, 1),
        "ckpt": str(args.ckpt),
        "frame_label": "clip label broadcast to frames (snn_hw_pipeline.py:319), same handicap as comparison/evaluate.py",
        "test_background_hours": round(float(test_nframes[test_labels == 0].sum()) * DT / 3600, 3),
        "budgets": {},
    }
    for budget in BUDGETS:
        chosen = choose_operating_point(val_report, budget)
        if chosen["feasible"]:
            test_report = stream_report(
                test_trains, test_labels, test_kinds, test_groups, n_frames=test_nframes,
                dt=DT, rules=(tuple(chosen["rule"]),), budgets=(budget,),
                refrac=REFRAC_FRAMES, n_boot=args.boot,
            )
        else:
            test_report = {budget: val_report[budget]}  # unused; report_on_test short-circuits on infeasible
        result["budgets"][str(budget)] = {"chosen_on_val": chosen, "test": report_on_test(test_report, chosen, budget)}
        print(f"  budget {budget} FA/h -> {result['budgets'][str(budget)]['test']}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "snn.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"written {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
