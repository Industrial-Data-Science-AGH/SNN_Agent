#!/usr/bin/env python3
"""Train a 3 -> 10 -> 1 LIF SNN on positive/negative audio data."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import wave
from dataclasses import dataclass
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


def compute_max_vals(pos_dir: str, neg_dir: str, sr: int, cfg: EncoderConfig, segment_sec: float, segment_strategy: str, max_files: int, seed: int) -> np.ndarray:
    """Compute max values for normalization from actual training data."""
    print("Computing max_vals from training data...")
    max_peak = 0.0
    max_mean = 0.0
    max_std = 0.0
    
    pos_files = []
    neg_files = []
    for dirpath, _, filenames in os.walk(pos_dir):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                pos_files.append(os.path.join(dirpath, filename))
    for dirpath, _, filenames in os.walk(neg_dir):
        for filename in filenames:
            if filename.lower().endswith(".wav"):
                neg_files.append(os.path.join(dirpath, filename))
    
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
        max_peak = max(max_peak, float(feats[:, 0].max()))
        max_mean = max(max_mean, float(feats[:, 1].max()))
        max_std = max(max_std, float(feats[:, 2].max()))
    
    print(f"Computed max_vals: peak={max_peak:.2f}, mean={max_mean:.2f}, std={max_std:.2f}")
    return np.array([max_peak, max_mean, max_std], dtype=np.float32)


# ------------------------------
# Spike encoder
# ------------------------------

@dataclass
class EncoderConfig:
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
    encoder_mode: str = "rate"
    dt_ms: float = 1.0
    
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
            "max_peak_val": self.max_peak_val,
            "max_mean_val": self.max_mean_val,
            "max_std_val": self.max_std_val,
        }
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved encoder max_vals to {json_path}")


def extract_hpf_features(x: np.ndarray, sr: int, cfg: EncoderConfig) -> np.ndarray:
    frame_size = int(round(cfg.frame_window_ms * 1e-3 * sr))
    if frame_size < 1:
        frame_size = 1

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
    def __init__(self, beta: float = 0.95) -> None:
        super().__init__()
        self.n_input = 3
        self.n_hidden = 10
        self.n_output = 1

        self.w_input_hidden = nn.Parameter(torch.randn(self.n_input, self.n_hidden) * 0.1 + 0.5)
        self.w_hidden_output = nn.Parameter(torch.randn(self.n_hidden, self.n_output) * 0.1 + 0.5)

        # Vectorized LIF neurons - one per layer, processes entire batch at once
        self.lif_input = snn.Leaky(beta=beta, threshold=0.5, learn_beta=True, learn_threshold=True)
        self.lif_hidden = snn.Leaky(beta=beta, threshold=0.5, learn_beta=True, learn_threshold=True)
        self.lif_output = snn.Leaky(beta=beta, threshold=0.5, learn_beta=True, learn_threshold=True)

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

        # Initialize memory states for vectorized neurons
        mem_input = self.lif_input.init_leaky()
        mem_hidden = self.lif_hidden.init_leaky()
        mem_output = self.lif_output.init_leaky()

        # Record spikes over time for analysis
        spk_input_rec = torch.zeros((n_timesteps, batch_size, self.n_input), device=x.device)
        spk_hidden_rec = torch.zeros((n_timesteps, batch_size, self.n_hidden), device=x.device)
        spk_output_rec = torch.zeros((n_timesteps, batch_size, self.n_output), device=x.device)

        for t in range(n_timesteps):
            # Input layer: [batch, n_input]
            cur_input = x[:, :, t]
            spk_input, mem_input = self.lif_input(cur_input, mem_input)
            spk_input_rec[t] = spk_input

            # Hidden layer: [batch, n_hidden]
            cur_hidden = torch.matmul(spk_input, self.w_input_hidden)  # [batch, n_hidden]
            spk_hidden, mem_hidden = self.lif_hidden(cur_hidden, mem_hidden)
            spk_hidden_rec[t] = spk_hidden

            # Output layer: [batch, n_output]
            cur_output = torch.matmul(spk_hidden, self.w_hidden_output)  # [batch, n_output]
            spk_output, mem_output = self.lif_output(cur_output, mem_output)
            spk_output_rec[t] = spk_output

        # Compute trigger (mean spike rate over time)
        trigger = spk_output_rec.mean(dim=0)  # [batch, n_output]

        out = {
            "input": spk_input_rec.permute(1, 2, 0),  # [batch, n_input, time]
            "hidden": spk_hidden_rec.permute(1, 2, 0),  # [batch, n_hidden, time]
            "output": spk_output_rec.permute(1, 2, 0),  # [batch, n_output, time]
        }
        return trigger, out


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
        self.cfg = cfg
        self.sr = sr
        self.segment_sec = segment_sec
        self.segment_strategy = segment_strategy

        pos_files = self._list_wav_files(pos_dir)
        neg_files = self._list_wav_files(neg_dir)
        random.Random(seed).shuffle(pos_files)
        random.Random(seed).shuffle(neg_files)

        if max_files > 0:
            pos_files = pos_files[:max_files]
            neg_files = neg_files[:max_files]

        target_steps = int(round(segment_sec * 1000.0 / cfg.dt_ms))
        for path in pos_files:
            x, _ = load_wav_mono(path, sr)
            x = pick_segment(x, sr, segment_sec, segment_strategy)
            spikes = encode_spikes(x, sr, cfg)
            spikes = fix_spike_length(spikes, target_steps)
            self.samples.append((spikes, 1))

        for path in neg_files:
            x, _ = load_wav_mono(path, sr)
            x = pick_segment(x, sr, segment_sec, segment_strategy)
            spikes = encode_spikes(x, sr, cfg)
            spikes = fix_spike_length(spikes, target_steps)
            self.samples.append((spikes, 0))

        random.Random(seed).shuffle(self.samples)

    def _list_wav_files(self, root_dir: str) -> List[str]:
        result: List[str] = []
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(".wav"):
                    result.append(os.path.join(dirpath, filename))
        return result

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


# ------------------------------
# Training
# ------------------------------


def evaluate(model: GlassBreakSNN, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    tp = fp = tn = fn = 0
    criterion = nn.BCELoss()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            trigger, _ = model(x)
            pred = (trigger >= 0.5).float()
            correct += (pred == y).sum().item()
            total += y.numel()
            loss_sum += criterion(trigger, y).item() * y.size(0)
            
            # Calculate TP, TN, FP, FN for F1, precision, recall
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
        'accuracy': acc,
        'loss': loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def train(
    model: GlassBreakSNN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> None:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            optimizer.zero_grad()
            trigger, _ = model(x)
            loss = criterion(trigger, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * y.size(0)

        train_metrics = evaluate(model, train_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:03d}: loss={train_metrics['loss']:.4f}, acc={train_metrics['accuracy']:.3f}, "
            f"f1={train_metrics['f1']:.3f} | "
            f"val_loss={val_metrics['loss']:.4f}, val_acc={val_metrics['accuracy']:.3f}, "
            f"val_f1={val_metrics['f1']:.3f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 3-10-1 LIF SNN on positive/negative WAV data")
    parser.add_argument("--positive-dir", default="positive/positive", help="Root directory for positive WAV files")
    parser.add_argument("--negative-dir", default="negative/negative", help="Root directory for negative WAV files")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--segment-sec", type=float, default=0.5)
    parser.add_argument("--segment-strategy", choices=["start", "center", "max_energy"], default="max_energy")
    parser.add_argument("--encoder-mode", choices=["rate", "ttfs"], default="rate")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--max-files", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    cfg = EncoderConfig(encoder_mode=args.encoder_mode)

    # Compute max_vals from actual training data for normalization
    max_vals_path = os.path.join("experiments", "max_vals.json")
    max_vals = compute_max_vals(
        args.positive_dir, args.negative_dir, args.sample_rate, cfg,
        args.segment_sec, args.segment_strategy, args.max_files, args.seed
    )
    cfg.max_peak_val = float(max_vals[0])
    cfg.max_mean_val = float(max_vals[1])
    cfg.max_std_val = float(max_vals[2])
    cfg.to_json(max_vals_path)
    
    # Recreate dataset with computed max_vals
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

    if len(dataset) == 0:
        raise SystemExit("No WAV files found in positive or negative directories")

    n_test = int(len(dataset) * args.test_ratio)
    n_train = len(dataset) - n_test
    train_set, test_set = torch.utils.data.random_split(dataset, [n_train, n_test], generator=torch.Generator().manual_seed(args.seed))

    # Calculate class weights for imbalanced dataset
    train_labels = [dataset.samples[idx][1] for idx in train_set.indices]
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / torch.from_numpy(np.sqrt(class_counts)).float()
    sample_weights = torch.tensor([class_weights[label] for label in train_labels]).float()
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    print(f"Dataset class distribution - Positive: {class_counts[1]}, Negative: {class_counts[0]}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_batch)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    model = GlassBreakSNN()
    train(model, train_loader, test_loader, device, epochs=args.epochs, lr=args.lr)

    save_path = os.path.join("experiments", "glassbreak_snn_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model state dict to {save_path}")
    
    # Print learned parameters for hardware mapping
    print("\n=== Hardware Parameter Mapping ===")
    beta_val = model.lif_hidden.beta.item()
    threshold_val = model.lif_hidden.threshold.item()
    tau_mem = 1.0 / (1.0 - beta_val) if beta_val < 1.0 else float('inf')
    # Map tau_mem to potentiometer range (0-255 typical for 8-bit potentiometer)
    # Assuming potentiometer range corresponds to tau_mem from ~1 to ~10
    pot_value = max(0, min(255, int(round((tau_mem - 1.0) / 10.0 * 255))))
    print(f"Hidden layer: beta={beta_val:.4f}, tau_mem={tau_mem:.4f}, pot_value={pot_value} (threshold={threshold_val:.4f})")
    print(f"Normalization params (for Arduino): MAX_PEAK={cfg.max_peak_val:.2f}, MAX_MEAN={cfg.max_mean_val:.2f}, MAX_STD={cfg.max_std_val:.2f}")


if __name__ == "__main__":
    main()
