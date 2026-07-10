#!/usr/bin/env python3
"""Diagnostic: check class separability of 3 candidate features before
committing them to the spike encoder / re-encoding the whole dataset.

Candidates (all computed per-frame, then pooled/averaged over the segment,
same style as the original peak/mean/std diagnostic):

  1. hf_ratio    - high-frequency energy ratio. Two single-pole high-pass
                   filters are run on the raw signal with different alpha
                   (a lower alpha => higher cutoff => narrower high-frequency
                   band; a higher alpha => lower cutoff => broader band).
                   ratio = narrow-high-band energy / broadband energy.
                   Glass break should skew this toward 1 (energy concentrated
                   at high frequency); low-pitched thuds/footsteps should
                   skew it toward 0.

  2. zcr         - zero-crossing rate of the (broadband) high-pass filtered
                   signal within each frame. A cheap, filter-free proxy for
                   dominant frequency content: more zero crossings per frame
                   roughly means higher-frequency content.

  3. crest       - crest factor (peak / RMS) of the broadband HPF signal per
                   frame. Impulsive, "tinkly" bursts (glass) should have a
                   higher peak-to-RMS ratio than continuous or smoothly
                   rising noise.

This script does NOT touch the spike encoder or the training loop. It's
meant purely to tell you, with a number, whether these are worth wiring in
before you go re-encode your whole dataset and retrain.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import wave
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# ------------------------------
# Audio I/O (same as training script)
# ------------------------------

def load_wav_mono(path: str, target_sr: int) -> Tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        nch = wf.getnchannels()
        sr = wf.getframerate()
        width = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if width == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
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
    xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(xq, xp, x).astype(np.float32)


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


def _list_wav_files(root_dir: str) -> List[str]:
    result: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                result.append(os.path.join(dirpath, filename))
    return result


# ------------------------------
# Candidate feature extraction
# ------------------------------

@dataclass
class AltEncoderConfig:
    frame_window_ms: float = 20.0
    hpf_alpha_broad: float = 0.99   # lower cutoff -> broadband high-pass
    hpf_alpha_narrow: float = 0.85  # higher cutoff -> narrow, high-frequency-only band
    eps: float = 1e-6


def extract_alt_features(x: np.ndarray, sr: int, cfg: AltEncoderConfig) -> np.ndarray:
    """Returns per-frame [hf_ratio, zcr, crest] features."""
    frame_size = int(round(cfg.frame_window_ms * 1e-3 * sr))
    if frame_size < 1:
        frame_size = 1

    x_adc = (x * 0.5 + 0.5) * 1023.0
    x_adc = x_adc.astype(np.float32)

    n_frames = int(math.ceil(len(x_adc) / frame_size))
    feats = np.zeros((n_frames, 3), dtype=np.float32)

    hp_broad = 0.0
    hp_narrow = 0.0
    prev_raw = 512.0
    idx = 0

    for f in range(n_frames):
        sum_sq_broad = 0.0
        sum_sq_narrow = 0.0
        max_abs_broad = 0.0
        zero_crossings = 0
        prev_sign = None
        count = 0

        for _ in range(frame_size):
            if idx >= len(x_adc):
                raw = prev_raw
            else:
                raw = x_adc[idx]

            hp_broad = cfg.hpf_alpha_broad * (hp_broad + raw - prev_raw)
            hp_narrow = cfg.hpf_alpha_narrow * (hp_narrow + raw - prev_raw)
            prev_raw = raw

            sum_sq_broad += hp_broad * hp_broad
            sum_sq_narrow += hp_narrow * hp_narrow
            abs_broad = abs(hp_broad)
            if abs_broad > max_abs_broad:
                max_abs_broad = abs_broad

            sign = hp_broad >= 0.0
            if prev_sign is not None and sign != prev_sign:
                zero_crossings += 1
            prev_sign = sign

            count += 1
            idx += 1

        if count == 0:
            count = 1

        energy_broad = sum_sq_broad / count
        energy_narrow = sum_sq_narrow / count
        rms_broad = math.sqrt(energy_broad)

        feats[f, 0] = energy_narrow / (energy_broad + cfg.eps)      # hf_ratio
        feats[f, 1] = zero_crossings / float(count)                 # zcr
        feats[f, 2] = max_abs_broad / (rms_broad + cfg.eps)          # crest factor

    return feats


def report_separability(pooled_features: List[Tuple[np.ndarray, int]]) -> None:
    pos = np.array([f for f, l in pooled_features if l == 1])
    neg = np.array([f for f, l in pooled_features if l == 0])
    names = ["hf_ratio", "zcr", "crest"]
    print("\n=== Candidate feature separability check (hf_ratio / zcr / crest) ===")
    for i, name in enumerate(names):
        pm, ps = pos[:, i].mean(), pos[:, i].std()
        nm, ns = neg[:, i].mean(), neg[:, i].std()
        pooled_std = math.sqrt((ps ** 2 + ns ** 2) / 2.0) + 1e-8
        d = abs(pm - nm) / pooled_std
        flag = "" if d > 0.5 else ("  <-- weak separation" if d < 0.2 else "  <-- small-to-medium")
        print(f"  {name:8s}: pos={pm:8.4f}±{ps:7.4f}  neg={nm:8.4f}±{ns:7.4f}  |d|={d:5.2f}{flag}")
    print("(|d| ~0.2 small, ~0.5 medium, ~0.8+ large effect size.)\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check separability of alternative acoustic features")
    parser.add_argument("--positive-dir", default="positive/positive")
    parser.add_argument("--negative-dir", default="negative/negative")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-sec", type=float, default=0.5)
    parser.add_argument("--segment-strategy", choices=["start", "center", "max_energy"], default="max_energy")
    parser.add_argument("--frame-window-ms", type=float, default=20.0)
    parser.add_argument("--hpf-alpha-broad", type=float, default=0.99)
    parser.add_argument("--hpf-alpha-narrow", type=float, default=0.85)
    parser.add_argument("--max-files", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AltEncoderConfig(
        frame_window_ms=args.frame_window_ms,
        hpf_alpha_broad=args.hpf_alpha_broad,
        hpf_alpha_narrow=args.hpf_alpha_narrow,
    )

    pos_files = _list_wav_files(args.positive_dir)
    neg_files = _list_wav_files(args.negative_dir)
    random.Random(args.seed).shuffle(pos_files)
    random.Random(args.seed).shuffle(neg_files)

    if args.max_files > 0:
        pos_files = pos_files[:args.max_files]
        neg_files = neg_files[:args.max_files]

    print(f"Loaded {len(pos_files)} positive / {len(neg_files)} negative files")

    pooled_features: List[Tuple[np.ndarray, int]] = []
    for path, label in [(p, 1) for p in pos_files] + [(n, 0) for n in neg_files]:
        x, _ = load_wav_mono(path, args.sample_rate)
        x = pick_segment(x, args.sample_rate, args.segment_sec, args.segment_strategy)
        feats = extract_alt_features(x, args.sample_rate, cfg)
        pooled_features.append((feats.mean(axis=0), label))

    if not pooled_features:
        raise SystemExit("No WAV files found in positive or negative directories")

    report_separability(pooled_features)


if __name__ == "__main__":
    main()