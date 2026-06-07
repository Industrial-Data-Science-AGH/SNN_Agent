"""
train_snn.py - Software training for a 15-neuron LIF SNN using spike-encoded audio.

Pipeline:
    1) Load WAV files from notebooks/dataset/{glass,negative}
    2) Encode spikes using snn_encoder_params_hpf (Peak/Mean/Std with HPF)
    3) Simulate a fixed 15-neuron topology (positive weights only)
    4) Random search + local refinement over grouped parameters
    5) Export best params to JSON for hardware pot mapping

Usage:
    python experiments/train_snn.py --dataset notebooks/dataset \
            --sample-rate 16000 --segment-sec 0.5 --trials 200 --refine 80
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import wave
from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np


# ------------------------------
# Audio I/O
# ------------------------------

def load_wav_mono(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        data /= 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32)
        data /= 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {width}")

    if nch > 1:
        data = data.reshape(-1, nch).mean(axis=1)

    if sr != target_sr:
        data = resample_linear(data, sr, target_sr)
        sr = target_sr

    return data, sr


def resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return x
    ratio = float(dst_sr) / float(src_sr)
    n_out = int(round(len(x) * ratio))
    if n_out < 2:
        return x[:1]
    xp = np.linspace(0.0, 1.0, num=len(x), endpoint=False)
    fp = x
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(xq, xp, fp).astype(np.float32)


def pick_segment(x: np.ndarray, sr: int, segment_sec: float, strategy: str) -> np.ndarray:
    if segment_sec <= 0:
        return x
    seg_len = int(segment_sec * sr)
    if seg_len >= len(x):
        return x
    if strategy == "start":
        return x[:seg_len]
    if strategy == "center":
        start = max(0, (len(x) - seg_len) // 2)
        return x[start:start + seg_len]
    if strategy == "max_energy":
        win = seg_len
        hop = max(1, seg_len // 4)
        best_e = -1.0
        best_i = 0
        for i in range(0, len(x) - win, hop):
            seg = x[i:i + win]
            e = float(np.mean(seg * seg))
            if e > best_e:
                best_e = e
                best_i = i
        return x[best_i:best_i + win]
    return x[:seg_len]


# ------------------------------
# Spike encoder (snn_encoder_params_hpf)
# ------------------------------

@dataclass
class EncoderConfig:
    # Match encoder/snn_encoder_params_hpf/snn_encoder_params_hpf.ino
    frame_window_ms: float = 20.0
    max_peak_val: float = 100.0
    max_mean_val: float = 20.0
    max_std_val: float = 20.0
    rc_min_rate_hz: float = 5.0
    rc_max_rate_hz: float = 200.0
    rc_noise_floor: float = 0.05
    ttfs_threshold: float = 0.10
    hpf_alpha: float = 0.99
    smooth_alpha: float = 0.3
    encoder_mode: str = "rate"  # "rate" | "ttfs"
    dt_ms: float = 1.0


def extract_hpf_features(x: np.ndarray, sr: int, cfg: EncoderConfig) -> np.ndarray:
    frame_size = int(round(cfg.frame_window_ms * 1e-3 * sr))
    if frame_size < 1:
        frame_size = 1

    # Scale [-1,1] audio to [0,1023] to emulate ADC input
    x_adc = (x * 0.5 + 0.5) * 1023.0
    x_adc = x_adc.astype(np.float32)

    n_frames = int(math.ceil(len(x_adc) / frame_size))
    feats = np.zeros((n_frames, 3), dtype=np.float32)

    hp_filtered = 0.0
    prev_raw = 512.0

    idx = 0
    for f in range(n_frames):
        max_ac = 0.0
        sum_ac = 0.0
        sum_sq = 0.0
        count = 0
        for _ in range(frame_size):
            if idx >= len(x_adc):
                raw = prev_raw
            else:
                raw = x_adc[idx]
            hp_filtered = cfg.hpf_alpha * (hp_filtered + raw - prev_raw)
            prev_raw = raw
            val = abs(hp_filtered)
            if val > max_ac:
                max_ac = val
            sum_ac += val
            sum_sq += val * val
            count += 1
            idx += 1

        if count == 0:
            count = 1
        mean_ac = sum_ac / float(count)
        std_ac = math.sqrt(sum_sq / float(count))
        feats[f, 0] = max_ac
        feats[f, 1] = mean_ac
        feats[f, 2] = std_ac

    return feats


def encode_spikes(x: np.ndarray, sr: int, cfg: EncoderConfig) -> np.ndarray:
    feats = extract_hpf_features(x, sr, cfg)
    n_frames = feats.shape[0]
    frame_steps = max(1, int(round(cfg.frame_window_ms / cfg.dt_ms)))
    n_steps = n_frames * frame_steps
    spikes = np.zeros((n_steps, 3), dtype=np.float32)

    smoothed = np.zeros(3, dtype=np.float32)
    norm_vals = np.zeros((n_frames, 3), dtype=np.float32)
    max_vals = np.array([cfg.max_peak_val, cfg.max_mean_val, cfg.max_std_val], dtype=np.float32)

    for i in range(n_frames):
        smoothed = cfg.smooth_alpha * feats[i] + (1.0 - cfg.smooth_alpha) * smoothed
        norm = np.minimum(smoothed / max_vals, 1.0)
        norm_vals[i] = norm

    last_spike_step = np.full(3, -1_000_000, dtype=np.int64)
    curr_isi_steps = np.zeros(3, dtype=np.int64)
    ttfs_armed = np.zeros(3, dtype=bool)
    ttfs_time_step = np.zeros(3, dtype=np.int64)

    for step in range(n_steps):
        frame_idx = step // frame_steps
        if step % frame_steps == 0:
            norm = norm_vals[frame_idx]
            if cfg.encoder_mode == "rate":
                for ch in range(3):
                    if norm[ch] < cfg.rc_noise_floor:
                        curr_isi_steps[ch] = 0
                    else:
                        rate_hz = cfg.rc_min_rate_hz + norm[ch] * (cfg.rc_max_rate_hz - cfg.rc_min_rate_hz)
                        isi_ms = 1000.0 / max(rate_hz, 1e-3)
                        curr_isi_steps[ch] = max(1, int(round(isi_ms / cfg.dt_ms)))
            else:
                for ch in range(3):
                    ttfs_armed[ch] = False
                    if norm[ch] < cfg.ttfs_threshold:
                        ttfs_time_step[ch] = -1
                    else:
                        delay_steps = int(round(frame_steps * (1.0 - norm[ch])))
                        ttfs_time_step[ch] = (frame_idx * frame_steps) + delay_steps

        for ch in range(3):
            if cfg.encoder_mode == "rate":
                if curr_isi_steps[ch] > 0 and (step - last_spike_step[ch]) >= curr_isi_steps[ch]:
                    spikes[step, ch] = 1.0
                    last_spike_step[ch] = step
            else:
                if (not ttfs_armed[ch]) and ttfs_time_step[ch] >= 0 and step >= ttfs_time_step[ch]:
                    spikes[step, ch] = 1.0
                    ttfs_armed[ch] = True

    return spikes


# ------------------------------
# SNN model (LIF) - 15 neurons
# ------------------------------

@dataclass
class NeuronParams:
    w1: float
    w2: float
    w3: float
    vmem: float
    v_leak: float
    tau_mem_ms: float
    tau_syn_ms: float


@dataclass
class NeuronSpec:
    name: str
    sources: Tuple[Tuple[str, int] | None, Tuple[str, int] | None, Tuple[str, int] | None]
    group: str


def build_topology() -> List[NeuronSpec]:
    # Inputs are ch0, ch1, ch2 from encoder
    specs = [
        NeuronSpec("N1", (("in", 0), ("in", 1), None), "transient"),
        NeuronSpec("N2", (("in", 0), ("in", 2), None), "transient"),
        NeuronSpec("N3", (("in", 1), ("in", 2), None), "ring"),
        NeuronSpec("N4", (("in", 2), ("in", 1), None), "ring"),
        NeuronSpec("N5", (("in", 1), ("in", 1), ("in", 2)), "rate"),
        NeuronSpec("N6", (("n", 0), ("n", 2), None), "coinc"),
        NeuronSpec("N7", (("n", 1), ("n", 3), None), "coinc"),
        NeuronSpec("N8", (("n", 5), ("n", 6), ("n", 4)), "consensus"),
        NeuronSpec("N9", (("n", 2), ("n", 4), None), "validator"),
        NeuronSpec("N10", (("n", 7), ("n", 8), None), "decision"),
        NeuronSpec("N11", (("n", 9), None, None), "shaper"),
        NeuronSpec("N12", (None, None, None), "spare"),
        NeuronSpec("N13", (None, None, None), "spare"),
        NeuronSpec("N14", (None, None, None), "spare"),
        NeuronSpec("N15", (None, None, None), "spare"),
    ]
    return specs


def run_snn(spikes_in: np.ndarray, params: List[NeuronParams], specs: List[NeuronSpec], dt_ms: float) -> np.ndarray:
    n_steps = spikes_in.shape[0]
    n_neurons = len(specs)
    v = np.array([p.vmem for p in params], dtype=np.float32)
    syn = np.zeros((n_neurons, 3), dtype=np.float32)
    out = np.zeros((n_steps, n_neurons), dtype=np.float32)

    for t in range(n_steps):
        prev = out[t - 1] if t > 0 else np.zeros(n_neurons, dtype=np.float32)
        for i, spec in enumerate(specs):
            p = params[i]
            # decay synaptic currents
            if p.tau_syn_ms > 0:
                decay = math.exp(-dt_ms / p.tau_syn_ms)
            else:
                decay = 0.0
            syn[i] *= decay

            # add new spikes to synapses
            for k, src in enumerate(spec.sources):
                if src is None:
                    continue
                stype, sidx = src
                spk = 0.0
                if stype == "in":
                    spk = spikes_in[t, sidx]
                else:
                    spk = prev[sidx]
                if spk > 0.5:
                    if k == 0:
                        syn[i, k] += p.w1
                    elif k == 1:
                        syn[i, k] += p.w2
                    else:
                        syn[i, k] += p.w3

            # membrane update
            if p.tau_mem_ms > 0:
                dv = (dt_ms / p.tau_mem_ms) * (p.vmem - v[i])
            else:
                dv = 0.0
            v[i] = v[i] + dv - (p.v_leak * dt_ms) + syn[i].sum()

            if v[i] >= 1.0:
                out[t, i] = 1.0
                v[i] = p.vmem

    return out


# ------------------------------
# Training / search
# ------------------------------

GROUPS = ["transient", "ring", "rate", "coinc", "consensus", "validator", "decision", "shaper", "spare"]

BOUNDS = {
    "w": (0.05, 1.0),
    "vmem": (0.05, 0.6),
    "v_leak": (0.0, 0.05),
    "tau_mem_ms": (5.0, 400.0),
    "tau_syn_ms": (2.0, 80.0),
}


def sample_group_params(rng: np.random.Generator, group: str) -> NeuronParams:
    wmin, wmax = BOUNDS["w"]
    vmin, vmax = BOUNDS["vmem"]
    lmin, lmax = BOUNDS["v_leak"]
    tmin, tmax = BOUNDS["tau_mem_ms"]
    smin, smax = BOUNDS["tau_syn_ms"]

    # group-specific bias
    if group == "transient":
        tmin, tmax = 5.0, 40.0
        smin, smax = 2.0, 20.0
        lmin, lmax = 0.01, 0.05
    elif group == "ring":
        tmin, tmax = 40.0, 200.0
        smin, smax = 8.0, 40.0
        lmin, lmax = 0.0, 0.02
    elif group == "shaper":
        tmin, tmax = 80.0, 300.0
        smin, smax = 20.0, 80.0

    return NeuronParams(
        w1=float(rng.uniform(wmin, wmax)),
        w2=float(rng.uniform(wmin, wmax)),
        w3=float(rng.uniform(0.0, wmax)),
        vmem=float(rng.uniform(vmin, vmax)),
        v_leak=float(rng.uniform(lmin, lmax)),
        tau_mem_ms=float(rng.uniform(tmin, tmax)),
        tau_syn_ms=float(rng.uniform(smin, smax)),
    )


def expand_params(specs: List[NeuronSpec], group_params: Dict[str, NeuronParams]) -> List[NeuronParams]:
    params = []
    for spec in specs:
        gp = group_params.get(spec.group)
        if gp is None:
            gp = group_params["spare"]
        params.append(gp)
    return params


def evaluate(params: List[NeuronParams], specs: List[NeuronSpec], dataset: List[Tuple[np.ndarray, int]], dt_ms: float, out_idx: int) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    latencies = []
    for spikes, label in dataset:
        out = run_snn(spikes, params, specs, dt_ms)
        out_spikes = out[:, out_idx]
        detected = out_spikes.sum() > 0

        if detected and label == 1:
            tp += 1
            first = np.argmax(out_spikes > 0)
            latencies.append(first * dt_ms)
        elif detected and label == 0:
            fp += 1
        elif (not detected) and label == 0:
            tn += 1
        else:
            fn += 1

    total = tp + tn + fp + fn
    acc = (tp + tn) / max(1, total)
    fnr = fn / max(1, (tp + fn))
    fpr = fp / max(1, (fp + tn))
    latency = float(np.mean(latencies)) if latencies else 999.0

    score = 5.0 * fnr + 2.0 * fpr + 0.05 * (latency / 100.0)
    return {"acc": acc, "fnr": fnr, "fpr": fpr, "latency_ms": latency, "score": score}


def build_dataset(glass_dir: str, neg_dir: str, sr: int, cfg: EncoderConfig, segment_sec: float, strategy: str, max_files: int, seed: int) -> List[Tuple[np.ndarray, int]]:
    rng = random.Random(seed)
    glass_files = [os.path.join(glass_dir, f) for f in os.listdir(glass_dir) if f.lower().endswith(".wav")]
    neg_files = [os.path.join(neg_dir, f) for f in os.listdir(neg_dir) if f.lower().endswith(".wav")]
    rng.shuffle(glass_files)
    rng.shuffle(neg_files)
    if max_files > 0:
        glass_files = glass_files[:max_files]
        neg_files = neg_files[:max_files]

    dataset = []
    for path in glass_files:
        x, _ = load_wav_mono(path, sr)
        x = pick_segment(x, sr, segment_sec, strategy)
        spikes = encode_spikes(x, sr, cfg)
        dataset.append((spikes, 1))

    for path in neg_files:
        x, _ = load_wav_mono(path, sr)
        x = pick_segment(x, sr, segment_sec, strategy)
        spikes = encode_spikes(x, sr, cfg)
        dataset.append((spikes, 0))

    rng.shuffle(dataset)
    return dataset


def split_dataset(dataset: List[Tuple[np.ndarray, int]], test_ratio: float, seed: int):
    rng = random.Random(seed)
    rng.shuffle(dataset)
    n_test = int(len(dataset) * test_ratio)
    return dataset[n_test:], dataset[:n_test]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="notebooks/dataset", help="Path with glass/negative folders")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-sec", type=float, default=0.5)
    parser.add_argument("--segment-strategy", default="max_energy", choices=["start", "center", "max_energy"])
    parser.add_argument("--encoder-mode", default="rate", choices=["rate", "ttfs"])
    parser.add_argument("--frame-window-ms", type=float, default=20.0)
    parser.add_argument("--max-peak-val", type=float, default=100.0)
    parser.add_argument("--max-mean-val", type=float, default=20.0)
    parser.add_argument("--max-std-val", type=float, default=20.0)
    parser.add_argument("--rc-min-rate-hz", type=float, default=5.0)
    parser.add_argument("--rc-max-rate-hz", type=float, default=200.0)
    parser.add_argument("--rc-noise-floor", type=float, default=0.05)
    parser.add_argument("--ttfs-threshold", type=float, default=0.10)
    parser.add_argument("--hpf-alpha", type=float, default=0.99)
    parser.add_argument("--smooth-alpha", type=float, default=0.3)
    parser.add_argument("--dt-ms", type=float, default=1.0)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--refine", type=int, default=80)
    parser.add_argument("--max-files", type=int, default=120)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/snn_pot_targets.json")
    args = parser.parse_args()

    glass_dir = os.path.join(args.dataset, "glass")
    neg_dir = os.path.join(args.dataset, "negative")
    if not (os.path.isdir(glass_dir) and os.path.isdir(neg_dir)):
        raise SystemExit("Dataset missing glass/negative directories")

    enc_cfg = EncoderConfig(
        frame_window_ms=args.frame_window_ms,
        max_peak_val=args.max_peak_val,
        max_mean_val=args.max_mean_val,
        max_std_val=args.max_std_val,
        rc_min_rate_hz=args.rc_min_rate_hz,
        rc_max_rate_hz=args.rc_max_rate_hz,
        rc_noise_floor=args.rc_noise_floor,
        ttfs_threshold=args.ttfs_threshold,
        hpf_alpha=args.hpf_alpha,
        smooth_alpha=args.smooth_alpha,
        encoder_mode=args.encoder_mode,
        dt_ms=args.dt_ms,
    )

    dataset = build_dataset(
        glass_dir, neg_dir, args.sample_rate, enc_cfg,
        args.segment_sec, args.segment_strategy, args.max_files, args.seed
    )
    train_set, test_set = split_dataset(dataset, args.test_ratio, args.seed)

    specs = build_topology()
    dt_ms = enc_cfg.dt_ms
    out_idx = 10  # N11 output shaper

    rng = np.random.default_rng(args.seed)
    best = None
    best_score = 1e9

    for _ in range(args.trials):
        group_params = {g: sample_group_params(rng, g) for g in GROUPS}
        params = expand_params(specs, group_params)
        metrics = evaluate(params, specs, train_set, dt_ms, out_idx)
        if metrics["score"] < best_score:
            best_score = metrics["score"]
            best = (group_params, metrics)

    # Local refinement
    if best is not None and args.refine > 0:
        group_params, _ = best
        for _ in range(args.refine):
            candidate = {}
            for g, p in group_params.items():
                jitter = 0.15
                candidate[g] = NeuronParams(
                    w1=float(np.clip(p.w1 * (1 + rng.uniform(-jitter, jitter)), *BOUNDS["w"])),
                    w2=float(np.clip(p.w2 * (1 + rng.uniform(-jitter, jitter)), *BOUNDS["w"])),
                    w3=float(np.clip(p.w3 * (1 + rng.uniform(-jitter, jitter)), 0.0, BOUNDS["w"][1])),
                    vmem=float(np.clip(p.vmem * (1 + rng.uniform(-jitter, jitter)), *BOUNDS["vmem"])),
                    v_leak=float(np.clip(p.v_leak * (1 + rng.uniform(-jitter, jitter)), *BOUNDS["v_leak"])),
                    tau_mem_ms=float(np.clip(p.tau_mem_ms * (1 + rng.uniform(-jitter, jitter)), *BOUNDS["tau_mem_ms"])),
                    tau_syn_ms=float(np.clip(p.tau_syn_ms * (1 + rng.uniform(-jitter, jitter)), *BOUNDS["tau_syn_ms"])),
                )
            params = expand_params(specs, candidate)
            metrics = evaluate(params, specs, train_set, dt_ms, out_idx)
            if metrics["score"] < best_score:
                best_score = metrics["score"]
                best = (candidate, metrics)

    if best is None:
        raise SystemExit("Training failed to find params")

    group_params, train_metrics = best
    final_params = expand_params(specs, group_params)
    test_metrics = evaluate(final_params, specs, test_set, dt_ms, out_idx)

    export = {
        "encoder": {
            "frame_window_ms": enc_cfg.frame_window_ms,
            "max_peak_val": enc_cfg.max_peak_val,
            "max_mean_val": enc_cfg.max_mean_val,
            "max_std_val": enc_cfg.max_std_val,
            "rc_min_rate_hz": enc_cfg.rc_min_rate_hz,
            "rc_max_rate_hz": enc_cfg.rc_max_rate_hz,
            "rc_noise_floor": enc_cfg.rc_noise_floor,
            "ttfs_threshold": enc_cfg.ttfs_threshold,
            "hpf_alpha": enc_cfg.hpf_alpha,
            "smooth_alpha": enc_cfg.smooth_alpha,
            "encoder_mode": enc_cfg.encoder_mode,
            "dt_ms": enc_cfg.dt_ms,
            "sample_rate": args.sample_rate,
            "segment_sec": args.segment_sec,
            "segment_strategy": args.segment_strategy,
        },
        "topology": [{"name": s.name, "sources": s.sources, "group": s.group} for s in specs],
        "group_params": {g: vars(p) for g, p in group_params.items()},
        "neurons": [vars(p) for p in final_params],
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }

    with open(args.output, "w") as f:
        json.dump(export, f, indent=2)

    print("Saved:", args.output)
    print("Train:", train_metrics)
    print("Test: ", test_metrics)


if __name__ == "__main__":
    main()
