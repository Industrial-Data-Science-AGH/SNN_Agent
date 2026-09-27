#!/usr/bin/env python3
"""Write a golden replay of NeuronFrames from a real model package (task P3).

The dashboard needs frames to develop against before a device is streaming, and
hand-written ones drift away from what the runtime actually sends. This runs the
real integrator on a deterministic pseudo-random channel stream and dumps what
``LuiRuntime.snapshot()`` produced, so the demo data and the live data are the
same shape by construction.

The stream is synthetic, and every frame says so: with an uncalibrated package
``provenance`` stays ``simulated``. This is a replay for a UI, not a recording of
glass breaking.

    python -m snn_runtime.tools.make_demo_frames \
        --manifest tests/runtime/fixtures/lui8-v2-manifest.json \
        --frames 200 --every-us 50000 --out neuron-frames.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from contracts.validation import validate

from ..runtime import LuiRuntime
from ..telemetry import TelemetryFeed

BATCH_FRAMES = 10


def build(manifest: dict, *, frames: int, every_us: int, density: float, seed: int) -> list[dict]:
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    runtime.load(manifest)
    runtime.reset(epoch=1, source_time_us=0, device_id="demo-pi", session_id="demo-session")

    profile = manifest["encoder_profile"]
    fs, hop = profile["sample_rate_hz"], profile["hop_samples"]

    def grid(hop_index: int) -> int:
        """The device's timeline (rpi_agents/agent/batching.py), not dt_us.

        Generating demo frames on the dt_us grid produced batches the runtime
        correctly refuses, so the replay came out as untouched resting values.
        """
        return (hop_index * hop * 1_000_000 + fs // 2) // fs

    channels = [c["channel"] for c in sorted(profile["channel_map"], key=lambda c: c["index"])]
    rng = random.Random(seed)
    feed = TelemetryFeed(min_interval_us=every_us)

    out: list[dict] = []
    for index in range(0, frames, BATCH_FRAMES):
        spikes = [
            {"dt_us": grid(frame) - grid(index), "channel": channel}
            for frame in range(index, min(index + BATCH_FRAMES, frames))
            for channel in channels
            if rng.random() < density
        ]
        spikes.sort(key=lambda s: s["dt_us"])
        count = min(BATCH_FRAMES, frames - index)
        runtime.step(
            {
                "schema_version": "1.0",
                "request_id": f"demo-{index}",
                "device_id": "demo-pi",
                "session_id": "demo-session",
                "epoch": 1,
                "boot_id": "demo-boot",
                "batch_seq": index // BATCH_FRAMES,
                "encoder_hash": manifest["encoder_hash"],
                "source_start_us": grid(index),
                "source_end_us": grid(index + count),
                "spikes": spikes,
                "quality": {"dropped_events": 0, "adc_clipped": False},
            }
        )
        frame = feed.offer(runtime.snapshot())
        if frame is not None:
            validate("NeuronFrame", frame, manifest=manifest)
            out.append(frame)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=200, help="simulation frames to run (dt_us each)")
    parser.add_argument("--every-us", type=int, default=50_000, help="spacing of emitted frames in source time")
    parser.add_argument("--density", type=float, default=0.25, help="per-channel spike probability per frame")
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    frames = build(
        manifest, frames=args.frames, every_us=args.every_us, density=args.density, seed=args.seed
    )
    args.out.write_text(
        json.dumps({"schema_version": "1.0", "frames": frames}, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{len(frames)} frames -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
