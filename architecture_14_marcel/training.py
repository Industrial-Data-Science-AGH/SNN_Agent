import argparse
import csv
import json
import math
import os
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import snntorch as snn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


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
# Spike encoder
# ------------------------------
#
# Input channels are (hf_ratio, zcr, crest) instead of (peak, mean, std).
# These target frequency content and burst shape instead of raw loudness:
#
#   hf_ratio - energy of a narrow, high-cutoff filtered signal divided by
#              energy of a broader high-pass filtered signal.
#   zcr      - zero-crossing rate of the broadband filtered signal per frame.
#   crest    - peak-to-RMS ratio of the broadband filtered signal per frame.

@dataclass
class EncoderConfig:
    frame_window_ms: float = 20.0
    max_hf_ratio_val: float = 1.0
    max_zcr_val: float = 1.0
    max_crest_val: float = 10.0
    hpf_alpha_broad: float = 0.99   # lower cutoff -> broadband high-pass
    hpf_alpha_narrow: float = 0.85  # higher cutoff -> narrow, high-frequency-only band
    rc_min_rate_hz: float = 5.0
    rc_max_rate_hz: float = 200.0
    rc_noise_floor: float = 0.05
    ttfs_threshold: float = 0.10
    smooth_alpha: float = 0.3
    encoder_mode: str = "rate"
    dt_ms: float = 1.0
    eps: float = 1e-6

    @classmethod
    def from_json(cls, json_path: str) -> "EncoderConfig":
        """Load config from JSON file, using defaults for missing keys."""
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                data = json.load(f)
            cfg = cls()
            for key, value in data.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, value)
            return cfg
        return cls()

    def to_json(self, json_path: str) -> None:
        """Save config to JSON file."""
        data = {
            "max_hf_ratio_val": self.max_hf_ratio_val,
            "max_zcr_val": self.max_zcr_val,
            "max_crest_val": self.max_crest_val,
            "hpf_alpha_broad": self.hpf_alpha_broad,
            "hpf_alpha_narrow": self.hpf_alpha_narrow,
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved encoder max_vals to {json_path}")


def compute_max_vals(
    pos_dir: str,
    neg_dir: str,
    sr: int,
    cfg: EncoderConfig,
    segment_sec: float,
    segment_strategy: str,
    max_files: int,
    seed: int,
) -> np.ndarray:
    """Compute max values for normalization from actual training data."""
    print("Computing max_vals from training data...")
    max_hf_ratio = 0.0
    max_zcr = 0.0
    max_crest = 0.0

    pos_files = _list_wav_files(pos_dir)
    neg_files = _list_wav_files(neg_dir)

    random.Random(seed).shuffle(pos_files)
    random.Random(seed).shuffle(neg_files)

    if max_files > 0:
        pos_files = pos_files[:max_files]
        neg_files = neg_files[:max_files]

    all_files = pos_files + neg_files
    for path in all_files:
        x, _ = load_wav_mono(path, sr)
        x = pick_segment(x, sr, segment_sec, segment_strategy)
        feats = extract_hpf_features(x, sr, cfg)
        max_hf_ratio = max(max_hf_ratio, float(feats[:, 0].max()))
        max_zcr = max(max_zcr, float(feats[:, 1].max()))
        max_crest = max(max_crest, float(feats[:, 2].max()))

    print(f"Computed max_vals: hf_ratio={max_hf_ratio:.4f}, zcr={max_zcr:.4f}, crest={max_crest:.4f}")
    return np.array([max_hf_ratio, max_zcr, max_crest], dtype=np.float32)


def _list_wav_files(root_dir: str) -> List[str]:
    result: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                result.append(os.path.join(dirpath, filename))
    return result


def compute_max_vals_from_paths(
    paths: Sequence[str],
    sr: int,
    cfg: EncoderConfig,
    segment_sec: float,
    segment_strategy: str,
    max_files: int,
    seed: int,
) -> np.ndarray:
    """Compute max values from an explicit list of WAV files."""
    print("Computing max_vals from training data...")
    max_hf_ratio = 0.0
    max_zcr = 0.0
    max_crest = 0.0

    file_paths = list(paths)
    random.Random(seed).shuffle(file_paths)

    if max_files > 0:
        file_paths = file_paths[:max_files]

    for path in file_paths:
        x, _ = load_wav_mono(path, sr)
        x = pick_segment(x, sr, segment_sec, segment_strategy)
        feats = extract_hpf_features(x, sr, cfg)
        max_hf_ratio = max(max_hf_ratio, float(feats[:, 0].max()))
        max_zcr = max(max_zcr, float(feats[:, 1].max()))
        max_crest = max(max_crest, float(feats[:, 2].max()))

    print(f"Computed max_vals: hf_ratio={max_hf_ratio:.4f}, zcr={max_zcr:.4f}, crest={max_crest:.4f}")
    return np.array([max_hf_ratio, max_zcr, max_crest], dtype=np.float32)


def load_manifest_samples(manifest_path: str, repo_root: Optional[str], split: str) -> List[Tuple[str, int]]:
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest}")

    root = Path(repo_root).expanduser().resolve() if repo_root else manifest.parent.parent.parent
    rows: List[Tuple[str, int]] = []

    with manifest.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_split = (row.get("split") or "").strip().lower()
            if split != "all" and row_split and row_split != split:
                continue

            label_raw = (row.get("label") or "").strip().lower()
            if label_raw in {"positive", "1", "true"}:
                label = 1
            elif label_raw in {"negative", "0", "false"}:
                label = 0
            else:
                continue

            rel_path = (row.get("filepath") or "").strip()
            if not rel_path:
                continue
            path_obj = Path(rel_path)
            if not path_obj.is_absolute():
                path_obj = root / path_obj
            rows.append((str(path_obj.resolve()), label))

    if not rows:
        raise SystemExit(f"No rows matched split '{split}' in manifest {manifest}")
    return rows


def extract_hpf_features(x: np.ndarray, sr: int, cfg: EncoderConfig) -> np.ndarray:
    """Per-frame [hf_ratio, zcr, crest] features (name kept as
    extract_hpf_features for drop-in compatibility with the rest of the
    pipeline; it no longer returns peak/mean/std)."""
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

        feats[f, 0] = energy_narrow / (energy_broad + cfg.eps)   # hf_ratio
        feats[f, 1] = zero_crossings / float(count)              # zcr
        feats[f, 2] = max_abs_broad / (rms_broad + cfg.eps)       # crest

    return feats


def encode_spikes(x: np.ndarray, sr: int, cfg: EncoderConfig) -> np.ndarray:
    feats = extract_hpf_features(x, sr, cfg)
    n_frames = feats.shape[0]
    frame_steps = max(1, int(round(cfg.frame_window_ms / cfg.dt_ms)))
    n_steps = n_frames * frame_steps
    spikes = np.zeros((n_steps, 3), dtype=np.float32)

    smoothed = np.zeros(3, dtype=np.float32)
    norm_vals = np.zeros((n_frames, 3), dtype=np.float32)
    max_vals = np.array([cfg.max_hf_ratio_val, cfg.max_zcr_val, cfg.max_crest_val], dtype=np.float32)

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


def fix_spike_length(spikes: np.ndarray, target_steps: int) -> np.ndarray:
    if spikes.shape[0] == target_steps:
        return spikes
    result = np.zeros((target_steps, spikes.shape[1]), dtype=np.float32)
    n = min(spikes.shape[0], target_steps)
    result[:n] = spikes[:n]
    return result


# ------------------------------
# SNN model
# ------------------------------

class GlassBreakSNN(nn.Module):
    """3 -> n_hidden -> 1 LIF SNN.

    The output is read out from the *membrane potential* of the output
    neuron (averaged over time) rather than its binary spike train, so the
    loss gets a dense, continuous signal to differentiate through instead
    of a quantized 0/1 spike-rate average that can saturate at exactly 0.
    """

    def __init__(self, n_hidden: int = 10, beta: float = 0.95) -> None:
        super().__init__()
        self.n_input = 3
        self.n_hidden = n_hidden
        self.n_output = 1

        self.w_input_hidden = nn.Parameter(
            torch.randn(self.n_input, self.n_hidden) / math.sqrt(self.n_input)
        )
        self.w_hidden_output = nn.Parameter(
            torch.randn(self.n_hidden, self.n_output) / math.sqrt(self.n_hidden)
        )

        self.lif_input = snn.Leaky(beta=beta, threshold=0.5, learn_beta=True, learn_threshold=True)
        self.lif_hidden = snn.Leaky(beta=beta, threshold=0.5, learn_beta=True, learn_threshold=True)
        self.lif_output = snn.Leaky(beta=beta, threshold=0.5, learn_beta=True, learn_threshold=True)

    def clamp_dynamics(self) -> None:
        """Keep learned beta/threshold in physically valid ranges. beta and
        threshold are free parameters (learn_beta=True, learn_threshold=True)
        with no constraint, so Adam can push beta >= 1 (no longer 'leaky',
        can blow up over long unrolls) or threshold <= 0 (degenerate,
        fires every step). Call this after every optimizer.step()."""
        with torch.no_grad():
            for lif in (self.lif_input, self.lif_hidden, self.lif_output):
                lif.beta.clamp_(0.01, 0.999)
                lif.threshold.clamp_(0.05, 5.0)

    def forward(self, spike_input: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        if spike_input.dim() == 2:
            x = spike_input.unsqueeze(1).repeat(1, self.n_input, 1)
        elif spike_input.dim() == 3:
            if spike_input.shape[1] == self.n_input:
                x = spike_input
            elif spike_input.shape[2] == self.n_input:
                x = spike_input.permute(0, 2, 1)
            else:
                raise ValueError("Expected input with 3 channels in dim 1 or dim 2")
        else:
            raise ValueError("Expected input tensor with 2 or 3 dimensions")

        batch_size = x.shape[0]
        n_timesteps = x.shape[2]

        mem_input = self.lif_input.init_leaky()
        mem_hidden = self.lif_hidden.init_leaky()
        mem_output = self.lif_output.init_leaky()

        spk_input_rec = torch.zeros((n_timesteps, batch_size, self.n_input), device=x.device)
        spk_hidden_rec = torch.zeros((n_timesteps, batch_size, self.n_hidden), device=x.device)
        spk_output_rec = torch.zeros((n_timesteps, batch_size, self.n_output), device=x.device)
        mem_output_rec = torch.zeros((n_timesteps, batch_size, self.n_output), device=x.device)

        for t in range(n_timesteps):
            cur_input = x[:, :, t]
            spk_input, mem_input = self.lif_input(cur_input, mem_input)
            spk_input_rec[t] = spk_input

            cur_hidden = torch.matmul(spk_input, self.w_input_hidden)
            spk_hidden, mem_hidden = self.lif_hidden(cur_hidden, mem_hidden)
            spk_hidden_rec[t] = spk_hidden

            cur_output = torch.matmul(spk_hidden, self.w_hidden_output)
            spk_output, mem_output = self.lif_output(cur_output, mem_output)
            spk_output_rec[t] = spk_output
            mem_output_rec[t] = mem_output

        logits = mem_output_rec.mean(dim=0)  # [batch, n_output]
        spike_rate = spk_output_rec.mean(dim=0)  # [batch, n_output]

        out = {
            "input": spk_input_rec.permute(1, 2, 0),
            "hidden": spk_hidden_rec.permute(1, 2, 0),
            "output": spk_output_rec.permute(1, 2, 0),
            "mem_output": mem_output_rec.permute(1, 2, 0),
            "spike_rate": spike_rate,
        }
        return logits, out


# ------------------------------
# Dataset
# ------------------------------

class AudioSpikeDataset(Dataset):
    def __init__(
        self,
        pos_dir: str,
        neg_dir: str,
        sr: int,
        cfg: EncoderConfig,
        segment_sec: float,
        segment_strategy: str,
        max_files: int,
        seed: int,
    ):
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.pooled_features: List[Tuple[np.ndarray, int]] = []  # for the separability diagnostic
        self.cfg = cfg
        self.sr = sr
        self.segment_sec = segment_sec
        self.segment_strategy = segment_strategy

        pos_files = _list_wav_files(pos_dir)
        neg_files = _list_wav_files(neg_dir)
        random.Random(seed).shuffle(pos_files)
        random.Random(seed).shuffle(neg_files)

        if max_files > 0:
            pos_files = pos_files[:max_files]
            neg_files = neg_files[:max_files]

        target_steps = int(round(segment_sec * 1000.0 / cfg.dt_ms))
        for path, label in [(p, 1) for p in pos_files] + [(n, 0) for n in neg_files]:
            x, _ = load_wav_mono(path, sr)
            x = pick_segment(x, sr, segment_sec, segment_strategy)
            feats = extract_hpf_features(x, sr, cfg)
            self.pooled_features.append((feats.mean(axis=0), label))
            spikes = encode_spikes(x, sr, cfg)
            spikes = fix_spike_length(spikes, target_steps)
            self.samples.append((spikes, label))

        combined = list(zip(self.samples, self.pooled_features))
        if combined:
            random.Random(seed).shuffle(combined)
            self.samples, self.pooled_features = [list(t) for t in zip(*combined)]
        else:
            self.samples = []
            self.pooled_features = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        spikes, label = self.samples[idx]
        spikes_t = torch.from_numpy(spikes).transpose(0, 1)
        return spikes_t, label


class ManifestAudioSpikeDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        sr: int,
        cfg: EncoderConfig,
        segment_sec: float,
        segment_strategy: str,
        max_files: int,
        seed: int,
    ):
        self.samples: List[Tuple[np.ndarray, int]] = []
        self.pooled_features: List[Tuple[np.ndarray, int]] = []
        self.cfg = cfg
        self.sr = sr
        self.segment_sec = segment_sec
        self.segment_strategy = segment_strategy

        file_samples = list(samples)
        random.Random(seed).shuffle(file_samples)
        if max_files > 0:
            file_samples = file_samples[:max_files]

        target_steps = int(round(segment_sec * 1000.0 / cfg.dt_ms))
        for path, label in file_samples:
            x, _ = load_wav_mono(path, sr)
            x = pick_segment(x, sr, segment_sec, segment_strategy)
            feats = extract_hpf_features(x, sr, cfg)
            self.pooled_features.append((feats.mean(axis=0), label))
            spikes = encode_spikes(x, sr, cfg)
            spikes = fix_spike_length(spikes, target_steps)
            self.samples.append((spikes, label))

        combined = list(zip(self.samples, self.pooled_features))
        if combined:
            random.Random(seed).shuffle(combined)
            self.samples, self.pooled_features = [list(t) for t in zip(*combined)]
        else:
            self.samples = []
            self.pooled_features = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        spikes, label = self.samples[idx]
        spikes_t = torch.from_numpy(spikes).transpose(0, 1)
        return spikes_t, label


def collate_batch(batch: Sequence[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = torch.stack([item[0] for item in batch], dim=0)
    ys = torch.tensor([item[1] for item in batch], dtype=torch.float32)
    return xs, ys


def stratified_split(labels: Sequence[int], test_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    """Split indices into train/test while preserving class balance."""
    rng = random.Random(seed)
    by_class: dict = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    train_idx: List[int] = []
    test_idx: List[int] = []
    for label, idxs in by_class.items():
        idxs = idxs[:]
        rng.shuffle(idxs)
        n_test = int(round(len(idxs) * test_ratio))
        test_idx.extend(idxs[:n_test])
        train_idx.extend(idxs[n_test:])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def report_feature_separability(pooled_features: Sequence[Tuple[np.ndarray, int]]) -> None:
    """Cheap, dependency-free sanity check: do the pooled (hf_ratio, zcr,
    crest) features actually differ between classes?"""
    pos = np.array([f for f, l in pooled_features if l == 1])
    neg = np.array([f for f, l in pooled_features if l == 0])
    names = ["hf_ratio", "zcr", "crest"]
    print("\n=== Feature separability check (pooled hf_ratio/zcr/crest) ===")
    for i, name in enumerate(names):
        pm, ps = pos[:, i].mean(), pos[:, i].std()
        nm, ns = neg[:, i].mean(), neg[:, i].std()
        pooled_std = math.sqrt((ps ** 2 + ns ** 2) / 2.0) + 1e-8
        d = abs(pm - nm) / pooled_std
        flag = "" if d > 0.5 else "  <-- weak separation"
        print(f"  {name:8s}: pos={pm:8.4f}±{ps:7.4f}  neg={nm:8.4f}±{ns:7.4f}  |d|={d:5.2f}{flag}")
    print("(|d| ~0.2 small, ~0.5 medium, ~0.8+ large effect size.)\n")


# ------------------------------
# Training
# ------------------------------


def evaluate(model: GlassBreakSNN, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    tp = fp = tn = fn = 0
    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            logits, _ = model(x)
            pred = (logits >= 0.0).float()  # threshold at logit 0 == prob 0.5
            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += criterion(logits, y).item() * y.size(0)

            tp += ((pred == 1) & (y == 1)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()

    acc = correct / max(1, total)
    loss = loss_sum / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-8, precision + recall)

    return {
        "accuracy": acc,
        "loss": loss,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train(
    model: GlassBreakSNN,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    grad_clip: float,
) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    model.to(device)

    best_val_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            logits, _ = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # Keep beta/threshold in valid ranges after every step -- these
            # are unconstrained learnable parameters and can otherwise drift
            # to physically invalid values (beta >= 1) mid-training, which
            # destabilizes the hidden layer and shows up as erratic,
            # non-monotonic loss across epochs.
            model.clamp_dynamics()

        train_metrics = evaluate(model, train_loader, device)
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:03d}: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.3f}, "
                f"f1={train_metrics['f1']:.3f} | "
                f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.3f}, "
                f"val_f1={val_metrics['f1']:.3f}"
            )
            print(
                f"  Confusion matrix (train): TN={train_metrics['tn']} FP={train_metrics['fp']} "
                f"FN={train_metrics['fn']} TP={train_metrics['tp']}"
            )
            print(
                f"  Confusion matrix (val): TN={val_metrics['tn']} FP={val_metrics['fp']} "
                f"FN={val_metrics['fn']} TP={val_metrics['tp']}"
            )
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            print(
                f"Epoch {epoch:03d}: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.3f}, "
                f"f1={train_metrics['f1']:.3f} | no validation split"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best checkpoint by val_f1={best_val_f1:.3f}")

    return {"best_val_f1": best_val_f1}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3-N-1 LIF SNN on positive/negative WAV data")
    parser.add_argument("--positive-dir", default="positive/positive", help="Root directory for positive WAV files")
    parser.add_argument("--negative-dir", default="negative/negative", help="Root directory for negative WAV files")
    parser.add_argument("--manifest", default=None, help="Path to a combined dataset manifest CSV produced by build_combined_dataset.py")
    parser.add_argument("--repo-root", default=None, help="Repository root used to resolve relative paths from the manifest")
    parser.add_argument("--manifest-split", choices=["train", "val", "test", "all"], default="train",
                         help="Which split from the manifest to use for training")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-sec", type=float, default=0.5)
    parser.add_argument("--segment-strategy", choices=["start", "center", "max_energy"], default="max_energy")
    parser.add_argument("--encoder-mode", choices=["rate", "ttfs"], default="rate")
    parser.add_argument("--hpf-alpha-broad", type=float, default=0.99)
    parser.add_argument("--hpf-alpha-narrow", type=float, default=0.85)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Max grad norm; 0 disables clipping")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                         help="Fraction of dataset held out for test; 0 means train on full dataset")
    parser.add_argument("--max-files", type=int, default=-1, help="Max number of files to load from each directory; -1 means no limit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    cfg = EncoderConfig(
        encoder_mode=args.encoder_mode,
        hpf_alpha_broad=args.hpf_alpha_broad,
        hpf_alpha_narrow=args.hpf_alpha_narrow,
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "experiments")
    os.makedirs(output_dir, exist_ok=True)
    max_vals_path = os.path.join(output_dir, "max_vals.json")

    if args.manifest:
        manifest_samples = load_manifest_samples(args.manifest, args.repo_root, args.manifest_split)
        file_paths = [path for path, _ in manifest_samples]
        max_vals = compute_max_vals_from_paths(
            file_paths, args.sample_rate, cfg,
            args.segment_sec, args.segment_strategy, args.max_files, args.seed
        )
        dataset = ManifestAudioSpikeDataset(
            manifest_samples,
            sr=args.sample_rate,
            cfg=cfg,
            segment_sec=args.segment_sec,
            segment_strategy=args.segment_strategy,
            max_files=args.max_files,
            seed=args.seed,
        )
    else:
        max_vals = compute_max_vals(
            args.positive_dir, args.negative_dir, args.sample_rate, cfg,
            args.segment_sec, args.segment_strategy, args.max_files, args.seed
        )
        dataset = AudioSpikeDataset(
            pos_dir=args.positive_dir,
            neg_dir=args.negative_dir,
            sr=args.sample_rate,
            cfg=cfg,
            segment_sec=args.segment_sec,
            segment_strategy=args.segment_strategy,
            max_files=args.max_files,
            seed=args.seed,
        )
    cfg.max_hf_ratio_val = float(max_vals[0])
    cfg.max_zcr_val = float(max_vals[1])
    cfg.max_crest_val = float(max_vals[2])
    cfg.to_json(max_vals_path)

    if len(dataset) == 0:
        raise SystemExit("No WAV files found in positive or negative directories")

    report_feature_separability(dataset.pooled_features)

    all_labels = [label for _, label in dataset.samples]

    if args.test_ratio > 0.0:
        train_idx, test_idx = stratified_split(all_labels, args.test_ratio, args.seed)
        train_set = torch.utils.data.Subset(dataset, train_idx)
        test_set = torch.utils.data.Subset(dataset, test_idx)
        print(f"Using stratified test split: {len(test_idx)} test / {len(train_idx)} train samples")
    else:
        train_set = dataset
        test_set = None
        train_idx = list(range(len(dataset)))
        print("Using full dataset for training (no test split)")

    train_labels = [all_labels[i] for i in train_idx]
    class_counts = np.bincount(train_labels, minlength=2)

    # Inverse-FREQUENCY (not inverse sqrt) weighting. With a WeightedRandomSampler,
    # weighting each sample by 1/count(class) makes the *expected* fraction of
    # each class in a sampled batch exactly 50/50, regardless of how skewed the
    # raw counts are:
    #   frac_pos = (n_pos * 1/n_pos) / (n_pos * 1/n_pos + n_neg * 1/n_neg) = 0.5
    # The previous 1/sqrt(count) scheme under-corrects on heavily imbalanced
    # data (e.g. 33:1) and leaves the model seeing mostly-negative batches,
    # which is exactly the "always predict negative" collapse this fixes.
    class_weights = 1.0 / torch.from_numpy(np.maximum(class_counts, 1)).float()
    sample_weights = torch.tensor([class_weights[label] for label in train_labels]).float()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    pos_frac_expected = (
        class_counts[1] * class_weights[1].item()
        / (class_counts[1] * class_weights[1].item() + class_counts[0] * class_weights[0].item())
    ) if len(class_counts) > 1 else 0.0
    print(f"Dataset class distribution (train split) - Positive: {class_counts[1]}, Negative: {class_counts[0]}")
    print(f"Expected positive fraction per sampled batch (with balanced sampler): {pos_frac_expected:.3f}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_batch)
    test_loader = None if test_set is None else DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = GlassBreakSNN(n_hidden=args.hidden_size)
    train(model, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr, grad_clip=args.grad_clip)

    save_path = os.path.join(output_dir, "glassbreak_snn_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model state dict to {save_path}")

    print("\n=== Hardware Parameter Mapping ===")
    beta_val = model.lif_hidden.beta.item()
    threshold_val = model.lif_hidden.threshold.item()

    beta_val_clamped = min(max(beta_val, 0.0), 0.999)
    if beta_val != beta_val_clamped:
        print(f"NOTE: learned beta={beta_val:.4f} was outside valid (0,1) range, "
              f"clamped to {beta_val_clamped:.4f} for hardware mapping.")

    tau_mem = 1.0 / (1.0 - beta_val_clamped)
    pot_value = max(0, min(255, int(round((tau_mem - 1.0) / 10.0 * 255))))
    print(f"Hidden layer: beta={beta_val_clamped:.4f}, tau_mem={tau_mem:.4f}, "
          f"pot_value={pot_value} (threshold={threshold_val:.4f})")


if __name__ == "__main__":
    main()