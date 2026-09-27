"""Deterministic golden replay for the dashboard demo (task C3).

The frame file is large and fully derived, so it is generated on demand instead
of being committed. `ensure_neuron_frames()` writes it once if missing; the dev
harness calls it at startup and the tests get it via the same call. Being
deterministic (no RNG), the output is identical every time — a golden fixture.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

NEURON_FRAMES_PATH = Path(__file__).parent / "static" / "demo" / "neuron-frames.json"

_DT = 0.5
_DURATION = 50.0
_N = 8
_THRESHOLD = 1.0
_RESET = 0.0
_TAU = 8.0


def build_neuron_frames() -> dict:
    """The golden replay document (8 neurons, LIF dynamics, ~50 s)."""
    steps = int(_DURATION / _DT) + 1
    v = [0.15 * (j + 1) for j in range(_N)]
    frames = []
    for k in range(steps):
        t = round(k * _DT, 3)
        neurons = []
        for j in range(_N):
            drive = 1.10 + 0.45 * math.sin(0.25 * t + j * 0.8) + 0.10 * math.cos(0.6 * t + j)
            v[j] += (_DT / _TAU) * (-(v[j]) + drive * _TAU * 0.5)
            spiked = v[j] >= _THRESHOLD
            if spiked:
                v[j] = _RESET
            neurons.append({
                "neuron_id": f"n{j + 1}",
                "v_mem": round(_THRESHOLD if spiked else v[j], 4),
                "v_threshold": _THRESHOLD,
                "v_reset": _RESET,
                "spiked": spiked,
            })
        frames.append({
            "frame_seq": k,
            "t": t,
            "source_time_us": int(t * 1_000_000),
            "status": "running",
            "neurons": neurons,
        })
    return {
        "schema_version": "1.0",
        "demo": True,
        "session_id": "demo-session",
        "provenance": "demo",
        "potential_unit": "a.u.",
        "duration_s": _DURATION,
        "dt_s": _DT,
        "v_threshold": _THRESHOLD,
        "v_reset": _RESET,
        "calibration": {"status": "unverified", "label": "Unverified"},
        "neuron_ids": [f"n{j + 1}" for j in range(_N)],
        "frames": frames,
    }


def ensure_neuron_frames(path: Path = NEURON_FRAMES_PATH) -> Path:
    """Write the golden replay if it is not already on disk. Returns the path."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(build_neuron_frames(), separators=(",", ":")), encoding="utf-8")
    return path


if __name__ == "__main__":
    print(f"wrote {ensure_neuron_frames()}")
