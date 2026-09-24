#!/usr/bin/env python3
"""Turn an hw_*.json export into a ModelManifest. REFERENCE, not the exporter.

The production exporter belongs to the training pipeline (task M, Marcel). This
script exists so that the mapping written down in ``snn_runtime/MAPPING.md`` is
executable rather than prose, and so the runtime tests can be driven by a
manifest describing the network we actually trained instead of a demo fixture.

It deliberately refuses to invent the values the export does not carry: the
caller must pass them, because every one of them is a decision somebody owns
(see MAPPING.md, table "Czego nie ma w eksporcie").

    python -m snn_runtime.tools.make_reference_manifest \
        --export models/hw_v2_recallfa_s0.json \
        --checkpoint models/v2_recallfa_s0.pt \
        --artifact-path v2_recallfa_s0.pt \
        --model-id lui8-v2-recallfa-s0 \
        --encoder-sha <40 hex of encoder_twin.py at build time> \
        --out contracts-style-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from contracts.validation import content_hash

# Lu.i solder headers, in the order the export names them.
PORT_OF_HEADER = {"J1": 1, "J2": 2, "J3": 3}
SIGN_OF_SYMBOL = {"+": "excitatory", "-": "inhibitory"}

# The integrator implemented by this runtime; see snn_runtime/units.py for why
# it is named rather than assumed.
INTEGRATOR = "lui-order2-hard-reset"

# LuiNet resets hard to zero on spike and models no refractory period.
V_RESET = 0.0
REFRACTORY_US = 0

# Decoder defaults taken from snn_pipeline/stream_eval.py: rule k spikes within
# w frames, then DEFAULT_REFRAC = 500 frames of dead time.
DEFAULT_K = 1
DEFAULT_W_FRAMES = 1
DEFAULT_COOLDOWN_FRAMES = 500


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    export: dict[str, Any],
    *,
    model_id: str,
    status: str,
    profile_id: str,
    profile_provenance: str,
    encoder_implementation_sha: str,
    encoder_config_sha256: str,
    sample_rate_hz: int,
    hop_samples: int,
    pulse_width_us: int,
    artifact_path: str,
    artifact_sha256: str,
    topology_version: str,
    k: int,
    w_frames: int,
    cooldown_frames: int,
    runtime_version: str,
    training_run_id: str | None,
    dataset_manifest_hash: str | None,
    evaluation_hash: str | None,
    seed: int | None,
    calibration_id: str | None,
) -> dict[str, Any]:
    dt_us = int(round(export["dt_s"] * 1_000_000))
    channels: list[str] = export["channels"]

    encoder_profile = {
        "schema_version": "1.0",
        "profile_id": profile_id,
        "provenance": profile_provenance,
        "implementation_sha": encoder_implementation_sha,
        "config_sha256": encoder_config_sha256,
        "sample_rate_hz": sample_rate_hz,
        "hop_samples": hop_samples,
        "pulse_width_us": pulse_width_us,
        "clock": "source_monotonic_us",
        "channel_map": [
            {"channel": name, "index": i, "feature": name} for i, name in enumerate(channels)
        ],
    }

    neurons = []
    connections = []
    for board_id, board in export["boards"].items():
        neurons.append(
            {
                "neuron_id": board_id,
                "tau_mem_us": int(round(board["tau_mem_ms"] * 1000)),
                "tau_syn_us": int(round(board["tau_syn_ms"] * 1000)),
                "v_leak": board["v_leak"],
                "v_threshold": export["v_th"],
                "v_reset": V_RESET,
                "refractory_us": REFRACTORY_US,
                "potential_unit": "a.u.",
            }
        )
        for synapse in board["synapses"]:
            sign = SIGN_OF_SYMBOL[synapse["sign"]]
            weight = synapse["w_sim"]
            if (weight < 0) != (sign == "inhibitory"):
                raise ValueError(
                    f"{board_id}/{synapse['port']}: sign {synapse['sign']!r} disagrees with "
                    f"w_sim {weight}; the export is internally inconsistent"
                )
            connections.append(
                {
                    "source_kind": "channel" if synapse["from"] in channels else "neuron",
                    "source_id": synapse["from"],
                    "target_id": board_id,
                    "target_port": PORT_OF_HEADER[synapse["port"]],
                    "sign": sign,
                    # The contract stores magnitude only; polarity lives in "sign".
                    "weight": abs(weight),
                    "weight_unit": "a.u.",
                    "delay_us": 0,
                }
            )

    manifest = {
        "schema_version": "1.0",
        "model_id": model_id,
        "status": status,
        "encoder_hash": content_hash(encoder_profile),
        "encoder_profile": encoder_profile,
        "artifacts": [{"path": artifact_path, "sha256": artifact_sha256}],
        "runtime": {
            "implementation": "snn_runtime.LuiRuntime",
            "version": runtime_version,
            "integrator": INTEGRATOR,
            "dt_us": dt_us,
            "potential_unit": "a.u.",
        },
        "decoder": {
            "implementation": "k-of-w-with-cooldown",
            "version": "1.0",
            "window_us": w_frames * dt_us,
            "threshold": k,
            "cooldown_us": cooldown_frames * dt_us,
        },
        "topology": {
            "topology_version": topology_version,
            "mode": "hardware_compatible",
            "neurons": neurons,
            "connections": connections,
        },
        "provenance": {
            "training_run_id": training_run_id,
            "dataset_manifest_hash": dataset_manifest_hash,
            "checkpoint_hash": artifact_sha256,
            "evaluation_hash": evaluation_hash,
            "seed": seed,
            "calibration_id": calibration_id,
        },
    }
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True, help="hw_*.json produced by snn_hw_pipeline export")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--status", default="sandbox", choices=["demo", "sandbox", "evaluated_champion"])
    ap.add_argument("--artifact-path", required=True, help="path as it will appear in the package")
    ap.add_argument("--checkpoint", help="local .pt to hash; omit to pass --artifact-sha256")
    ap.add_argument("--artifact-sha256")
    ap.add_argument("--profile-id", required=True)
    ap.add_argument("--profile-provenance", default="reference", choices=["demo", "measured", "reference"])
    ap.add_argument("--encoder-sha", required=True, help="40 hex, commit of the encoder implementation")
    ap.add_argument("--encoder-config-sha256", required=True, help="sha256:<64 hex> of the encoder config")
    ap.add_argument("--sample-rate-hz", type=int, default=19231)
    ap.add_argument("--hop-samples", type=int, default=192)
    ap.add_argument("--pulse-width-us", type=int, default=500)
    ap.add_argument("--topology-version", required=True)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--w-frames", type=int, default=DEFAULT_W_FRAMES)
    ap.add_argument("--cooldown-frames", type=int, default=DEFAULT_COOLDOWN_FRAMES)
    ap.add_argument("--runtime-version", default="0.1.0")
    ap.add_argument("--training-run-id")
    ap.add_argument("--dataset-manifest-hash")
    ap.add_argument("--evaluation-hash")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--calibration-id")
    args = ap.parse_args()

    if not (args.checkpoint or args.artifact_sha256):
        ap.error("pass --checkpoint to hash a local file, or --artifact-sha256 directly")
    artifact_sha256 = args.artifact_sha256 or _sha256_file(Path(args.checkpoint))

    export = json.loads(Path(args.export).read_bytes().decode("utf-8", "replace"))
    manifest = build(
        export,
        model_id=args.model_id,
        status=args.status,
        profile_id=args.profile_id,
        profile_provenance=args.profile_provenance,
        encoder_implementation_sha=args.encoder_sha,
        encoder_config_sha256=args.encoder_config_sha256,
        sample_rate_hz=args.sample_rate_hz,
        hop_samples=args.hop_samples,
        pulse_width_us=args.pulse_width_us,
        artifact_path=args.artifact_path,
        artifact_sha256=artifact_sha256,
        topology_version=args.topology_version,
        k=args.k,
        w_frames=args.w_frames,
        cooldown_frames=args.cooldown_frames,
        runtime_version=args.runtime_version,
        training_run_id=args.training_run_id,
        dataset_manifest_hash=args.dataset_manifest_hash,
        evaluation_hash=args.evaluation_hash,
        seed=args.seed,
        calibration_id=args.calibration_id,
    )
    Path(args.out).write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
