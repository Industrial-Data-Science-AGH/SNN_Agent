#!/usr/bin/env python3
"""
fitness.py — funkcje oceny osobnika (genomu).

  * synth       — heurystyka bez torcha (test mechaniki GA).
  * RealFitness — proxy-trening GenomeNet, fitness = wybrana metryka z oceny
                  ZDARZENIOWEJ (dekoder "k spików D w oknie", agregacja do klipu).
                  Domyślnie AP (Average Precision) — bezprogowa, mniej szumna niż
                  progowany F1 po kilku epokach (#2). Można uśredniać po kilku
                  seedach (fitness_seeds) dla mniejszej wariancji.
                  __call__ przyjmuje `budget` (0..1) skalujący epoki (halving).

Dane: gotowy `_cache_*.npz` ładowany wprost (bez zapisu — omija WinError 5).
"""
from __future__ import annotations

import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

from genome import Genome, N_FEATURES


# ------------------------------------------------------------------ synth

def synth_fitness(g: Genome) -> float:
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
    """Zbiór okien wczytany wprost z `_cache_*.npz` (bez SpikeClips, bez zapisu)."""

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
    cache = os.path.join(data, f"_cache_T{T}_s{stride}.npz")
    if os.path.exists(cache):
        if limit:
            print(f"[data] uwaga: --limit ignorowany (uzywam gotowego cache {cache})",
                  flush=True)
        return CachedClips(cache)
    return SpikeClips(data, T=T, stride=stride, limit=limit)


# ------------------------------------------------------------------ real

class RealFitness:
    """Proxy-trening GenomeNet -> metryka zdarzeniowa jako fitness."""

    def __init__(self, arch_dir: str, data: str, val_data: Optional[str] = None,
                 limit: Optional[int] = None, epochs: int = 4, T: int = 200,
                 stride: int = 50, bs: int = 128, lr: float = 3e-3,
                 num_samples: int = 6000, val_cap: int = 3000, k: int = 2,
                 metric: str = "ap", fitness_seeds: int = 1,
                 pos_weight: float = 3.0, fanout_penalty: float = 0.01,
                 verbose: bool = True, seed: int = 0, device: Optional[str] = None):
        assert metric in ("ap", "clip_f1"), "metric: 'ap' albo 'clip_f1'"
        arch_dir = os.path.abspath(arch_dir)
        if arch_dir not in sys.path:
            sys.path.insert(0, arch_dir)
        import numpy as np
        import torch
        from snn_hw_pipeline import SpikeClips, split_by_file, make_sampler
        from torch.utils.data import DataLoader, Subset

        self.torch, self.np = torch, np
        self.DataLoader, self.Subset = DataLoader, Subset
        self.make_sampler = make_sampler
        self.epochs = epochs
        self.bs, self.lr, self.num_samples = bs, lr, num_samples
        self.pos_weight, self.fanout_penalty = pos_weight, fanout_penalty
        self.metric, self.fitness_seeds = metric, fitness_seeds
        self.verbose, self.seed, self.k = verbose, seed, k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        ds = load_clips(data, T, stride, limit, SpikeClips)
        if val_data:
            va = load_clips(val_data, T, stride, None, SpikeClips)
            self.tr_ds, self.tr_lab = ds, ds.lab
            vw, vl, vf = va.win, va.lab, va.fidx
        else:
            tr_idx, va_idx = split_by_file(ds, val_frac=0.2, seed=0)
            self.tr_ds, self.tr_lab = Subset(ds, tr_idx), ds.lab[tr_idx]
            vw, vl, vf = ds.win[va_idx], ds.lab[va_idx], ds.fidx[va_idx]

        rng = np.random.default_rng(1)
        if len(vl) > val_cap:
            sub = rng.choice(len(vl), size=val_cap, replace=False)
            vw, vl, vf = vw[sub], vl[sub], vf[sub]
        self.va_win, self.va_lab, self.va_fidx = vw, vl.astype(np.float32), vf
        print(f"[val] {len(self.va_lab)} okien / {len(np.unique(self.va_fidx))} klipów "
              f"(dekoder k={k}, metryka={metric}, seedy={fitness_seeds})", flush=True)

    def eval_events(self, model, k: Optional[int] = None):
        import net
        return net.genome_eval_events(model, self.va_win, self.va_lab,
                                      self.va_fidx, self.device, k or self.k)

    def train_once(self, g: Genome, epochs: int, seed: int):
        """Jeden trening (fresh init) + ocena. Zwraca (metrics, loss0, lossN)."""
        import net
        torch = self.torch
        torch.manual_seed(seed); self.np.random.seed(seed)
        model = net.GenomeNet(g, hw=None, quantize=False).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        dl_tr = self.DataLoader(self.tr_ds, batch_size=self.bs,
                                sampler=self.make_sampler(self.tr_lab, self.num_samples))
        loss0 = lossN = 0.0
        for ep in range(epochs):
            model.train(); model.set_mismatch(True)
            ls, nb = 0.0, 0
            for x, y in dl_tr:
                x, y = x.to(self.device), y.to(self.device)
                loss, _ = net.genome_loss(model(x), y, self.pos_weight, k_ref=self.k)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                ls += loss.item(); nb += 1
            model.set_mismatch(False)
            mean = ls / max(nb, 1)
            if ep == 0:
                loss0 = mean
            lossN = mean
        return self.eval_events(model), loss0, lossN

    def __call__(self, g: Genome, budget: float = 1.0) -> float:
        epochs = max(1, round(self.epochs * budget))
        aps, f1s, l0, lN = [], [], 0.0, 0.0
        for si in range(self.fitness_seeds):
            m, loss0, lossN = self.train_once(g, epochs, self.seed + si)
            aps.append(m["ap"]); f1s.append(m["clip_f1"])
            l0, lN = loss0, lossN
        ap = sum(aps) / len(aps)
        f1 = sum(f1s) / len(f1s)
        score = (ap if self.metric == "ap" else f1) \
            - self.fanout_penalty * _avg_fanout(g)
        # GUARD anty-miraż: sieć, która przy progu k nic nie wykrywa (clipF1==0),
        # jest bezużyteczna — AP potrafi wtedy fałszywie wyjść 1.0 przy prawie
        # milczącym neuronie. Takie rozwiązanie dostaje fitness ~0.
        dead = f1 <= 1e-9
        if dead:
            score = 0.0
        if self.verbose:
            std = (sum((a - ap) ** 2 for a in aps) / len(aps)) ** 0.5
            print(f"    [eval] {g.layer_sizes()} loss {l0:.3f}->{lN:.3f}  "
                  f"AP {ap:.3f}±{std:.3f}  clipF1 {f1:.3f}"
                  f"{'  [MARTWY->0]' if dead else ''} (ep{epochs}, "
                  f"seedy{self.fitness_seeds})", flush=True)
        return score


def _avg_fanout(g: Genome) -> float:
    counts = []
    for k in range(len(g.layers)):
        counts += [c for c in g.fanout_counts(k) if c > 0]
    return sum(counts) / max(len(counts), 1)


# ------------------------------------------------------------------ równolegle

_POOL_RF = None  # per-worker RealFitness


def _pool_worker_init(rf_kwargs: dict):
    """Inicjalizacja pracownika puli: własny RealFitness + 1 wątek torch.

    Kluczowy detal: każdy worker dostaje SWOJĄ instancję RealFitness (dane +
    trener) i `torch.set_num_threads(1)` — inaczej 18 procesów x domyślna liczba
    wątków torch by się wzajemnie dusiły (threading thrash). CPU, bo MPS jest
    dla tak małego SNN ~3x wolniejszy (pomierzone 2026-08-24).
    """
    global _POOL_RF
    import torch
    torch.set_num_threads(1)
    _POOL_RF = RealFitness(**rf_kwargs)


def _pool_worker_eval(arg):
    g, budget = arg
    return _POOL_RF(g, budget)


class ParallelFitness:
    """Procesowa pula wokół RealFitness — ewaluacja osobników na wielu rdzeniach.

    Pracownicy budują RealFitness raz (koszt: wczytanie cache danych ~sekunda),
    po czym oceniają genomy niezależnie. `batch` zachowuje kolejność — GA może
    ewaluować generację hurtem i wyjdzie z tym samym cache-mapa topologii.
    """

    def __init__(self, rf_kwargs: dict, max_workers: Optional[int] = None):
        self.rf_kwargs = dict(rf_kwargs)
        self.rf_kwargs["verbose"] = False
        self.max_workers = max_workers or os.cpu_count() or 1
        self._pool = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_pool_worker_init,
            initargs=(self.rf_kwargs,),
        )

    def batch(self, genomes, budget: float = 1.0) -> list:
        if not genomes:
            return []
        # chunksize=1: ocena jednego osobnika trwa ~5 s, więc IPC jest bez znaczenia —
        # a grupowanie w chunki (4) zostawiałoby tylko 2/18 workerów zajętych przy
        # partii rzędu 24 genomów. Kolejność wyników jest zachowana (map).
        return list(self._pool.map(_pool_worker_eval,
                                   [(g, budget) for g in genomes], chunksize=1))

    def __call__(self, g: Genome, budget: float = 1.0) -> float:
        return self.batch([g], budget)[0]

    def close(self):
        self._pool.shutdown(wait=True)
