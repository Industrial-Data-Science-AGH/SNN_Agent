#!/usr/bin/env python3
"""
fitness.py — funkcje oceny osobnika (genomu).

Dwa tryby:
  * real   — buduje GenomeNet, robi krótki proxy-trening (kilka epok) i zwraca F1
             z walidacji. Kosztowne, ale wierne.
  * synth  — szybka heurystyka bez torcha (pokrycie cech, fan-in, głębokość).
             Do testów mechaniki GA i preselekcji. NIE zastępuje realnego treningu.

Wczytywanie danych: jeśli w katalogu danych jest gotowy `_cache_T{T}_s{stride}.npz`
(zbudowany wcześniej przez SpikeClips), wczytujemy go BEZPOŚREDNIO — bez skanowania
CSV i bez zapisu. To omija WinError 5, gdy katalog danych jest tylko-do-odczytu,
a podpis cache (mtime plików) nie zgadza się po checkoutcie gita.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

from genome import Genome, N_FEATURES


# ------------------------------------------------------------------ tryb synth

def synth_fitness(g: Genome) -> float:
    """Tania, deterministyczna ocena topologii (bez treningu)."""
    if not g.is_valid():
        return 0.0
    cov = len(g.features_used()) / N_FEATURES
    fanins = [len(n) for layer in g.layers for n in layer]
    avg_fanin = sum(fanins) / len(fanins) / 3.0
    depth = g.n_hidden_layers()
    depth_score = math.exp(-0.5 * (depth - 2) ** 2)
    sizes = [len(l) for l in g.layers[:-1]] or [1]
    width_pen = sum(max(0, s - 4) for s in sizes) * 0.03
    return max(0.0, 0.5 * cov + 0.3 * avg_fanin + 0.2 * depth_score - width_pen)


# ------------------------------------------------------------------ dane

class CachedClips:
    """Zbiór okien wczytany wprost z `_cache_*.npz` (bez SpikeClips, bez zapisu).

    Zapewnia interfejs zgodny z SpikeClips: __len__, __getitem__ (float tensor,
    label), oraz atrybuty .lab / .fidx / .file_lab (dla split_by_file)."""

    def __init__(self, npz_path: str):
        import numpy as np
        z = np.load(npz_path, allow_pickle=False)
        self.win = z["win"]
        self.lab = z["lab"].astype(np.float32)
        self.fidx = z["fidx"] if "fidx" in z else np.zeros(len(self.lab), np.int32)
        self.file_lab = z["file_lab"] if "file_lab" in z else self.lab
        print(f"[data] cache-direct: {len(self.lab)} okien, pozytywnych "
              f"{int(self.lab.sum())} ({100*self.lab.mean():.1f}%) <- {npz_path}",
              flush=True)

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        import torch
        return torch.from_numpy(self.win[i]).float(), torch.tensor(self.lab[i])


def load_clips(data: str, T: int, stride: int, limit: Optional[int], SpikeClips):
    """Wczytaj gotowy cache jeśli jest; inaczej zbuduj przez SpikeClips."""
    cache = os.path.join(data, f"_cache_T{T}_s{stride}.npz")
    if os.path.exists(cache):
        if limit:
            print(f"[data] uwaga: --limit ignorowany (uzywam gotowego cache {cache})",
                  flush=True)
        return CachedClips(cache)
    return SpikeClips(data, T=T, stride=stride, limit=limit)


# ------------------------------------------------------------------ tryb real

class RealFitness:
    """Proxy-trening GenomeNet i zwrot F1 z walidacji jako fitness.
    Dane ładowane raz; potem każdy genom trenowany krótko."""

    def __init__(self, arch_dir: str, data: str, val_data: Optional[str] = None,
                 limit: Optional[int] = None, epochs: int = 4, T: int = 200,
                 stride: int = 50, bs: int = 128, lr: float = 3e-3,
                 num_samples: int = 6000, val_cap: int = 3000,
                 pos_weight: float = 3.0, fanout_penalty: float = 0.01,
                 seed: int = 0, device: Optional[str] = None):
        arch_dir = os.path.abspath(arch_dir)
        if arch_dir not in sys.path:
            sys.path.insert(0, arch_dir)
        import numpy as np
        import torch
        from snn_hw_pipeline import SpikeClips, split_by_file, make_sampler
        from torch.utils.data import DataLoader, Subset

        self.torch = torch
        self.np = np
        self.DataLoader, self.Subset = DataLoader, Subset
        self.cfg = dict(epochs=epochs, bs=bs, lr=lr, num_samples=num_samples,
                        val_cap=val_cap, pos_weight=pos_weight,
                        fanout_penalty=fanout_penalty, seed=seed)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ds = load_clips(data, T, stride, limit, SpikeClips)
        if val_data:
            va_ds = load_clips(val_data, T, stride, None, SpikeClips)
            self.tr_ds, self.tr_lab = ds, ds.lab
        else:
            tr_idx, va_idx = split_by_file(ds, val_frac=0.2, seed=0)
            self.tr_ds, self.tr_lab = Subset(ds, tr_idx), ds.lab[tr_idx]
            va_ds = Subset(ds, va_idx)
        rng = np.random.default_rng(1)
        if len(va_ds) > val_cap:
            sub = rng.choice(len(va_ds), size=val_cap, replace=False)
            va_ds = Subset(va_ds, sub)
        self.va_ds = va_ds
        self.make_sampler = make_sampler

    def __call__(self, g: Genome) -> float:
        import net
        torch, np = self.torch, self.np
        c = self.cfg
        torch.manual_seed(c["seed"]); np.random.seed(c["seed"])

        model = net.GenomeNet(g, hw=None, quantize=False).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
        dl_tr = self.DataLoader(self.tr_ds, batch_size=c["bs"],
                                sampler=self.make_sampler(self.tr_lab, c["num_samples"]))
        dl_va = self.DataLoader(self.va_ds, batch_size=256)

        best_f1 = 0.0
        for ep in range(c["epochs"]):
            model.train(); model.set_mismatch(True)
            for x, y in dl_tr:
                x, y = x.to(self.device), y.to(self.device)
                loss, _ = net.genome_loss(model(x), y, c["pos_weight"])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            model.set_mismatch(False)
            m = net.genome_eval_f1(model, dl_va, self.device)
            best_f1 = max(best_f1, m["f1"])

        return best_f1 - c["fanout_penalty"] * _avg_fanout(g)


def _avg_fanout(g: Genome) -> float:
    counts = []
    for k in range(len(g.layers)):
        counts += [c for c in g.fanout_counts(k) if c > 0]
    return sum(counts) / max(len(counts), 1)
