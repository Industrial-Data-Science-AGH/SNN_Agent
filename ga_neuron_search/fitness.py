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
                 test_data: Optional[str] = None,
                 limit: Optional[int] = None, epochs: int = 4, T: int = 200,
                 stride: int = 50, bs: int = 128, lr: float = 3e-3,
                 num_samples: int = 6000, val_cap: int = 3000, k: int = 1,
                 metric: str = "clip_f1", fitness_seeds: int = 1,
                 pos_weight: float = 3.0, fanout_penalty: float = 0.01,
                 feature_penalty: float = 0.005, channels_head: Optional[int] = None,
                 stream_budget: float = 6.0, stream_boot: int = 0,
                 verbose: bool = True, seed: int = 0, device: Optional[str] = None):
        assert metric in ("ap", "clip_f1", "recall_fa"), \
            "metric: 'ap' | 'clip_f1' | 'recall_fa' (recall @ budżet FA/h)"
        arch_dir = os.path.abspath(arch_dir)
        if arch_dir not in sys.path:
            sys.path.insert(0, arch_dir)
        # wspólny moduł metryki strumieniowej (recall @ FA/h) z snn_pipeline
        _pipe = os.path.join(os.path.dirname(arch_dir), "snn_pipeline")
        if os.path.isdir(_pipe) and _pipe not in sys.path:
            sys.path.insert(0, _pipe)
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
        self.feature_penalty = feature_penalty
        self.metric, self.fitness_seeds = metric, fitness_seeds
        self.verbose, self.seed, self.k = verbose, seed, k
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # metryka strumieniowa (recall @ budżet FA/h): potrzebny katalog splitu
        # z pełnymi klipami + files.csv (kind/group_id). Klipy val cache'owane.
        self.val_dir, self.data_dir, self.test_dir = val_data, data, test_data
        self.channels_head = channels_head
        self.stream_budget, self.stream_boot = stream_budget, stream_boot
        self._stream_clips = None      # cache klipów val (fitness)
        self._test_clips = None        # cache klipów test (raport końcowy)
        if metric == "recall_fa" and not val_data:
            raise ValueError("metric=recall_fa wymaga --val-data (katalog splitu z files.csv)")

        ds = load_clips(data, T, stride, limit, SpikeClips)
        # A/B selekcji cech: ogranicz do pierwszych `channels_head` kanałów (w
        # spikes_ext pierwsze 7 to HW) — pozwala porównać 7 HW vs 14 na TYM SAMYM
        # zbiorze/val/k (zmienia się tylko pula cech).
        if channels_head:
            ds.win = ds.win[..., :channels_head]
        if val_data:
            va = load_clips(val_data, T, stride, None, SpikeClips)
            if channels_head:
                va.win = va.win[..., :channels_head]
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

        # liczba kanałów encodera bierze się z danych (nie zaszyta na 7) — GA
        # selekcjonuje spośród nich. Nazwy: najpierw sidecar channels.json obok
        # danych (rozszerzone zbiory 14-kanałowe), potem pipeline, na końcu ch{i}.
        self.n_channels = int(self.va_win.shape[-1])
        self.channel_names = _load_channel_names(val_data or data, self.n_channels)

        print(f"[val] {len(self.va_lab)} okien / {len(np.unique(self.va_fidx))} klipów "
              f"(dekoder k={k}, metryka={metric}, seedy={fitness_seeds}, "
              f"kanały={self.n_channels})", flush=True)

    def eval_events(self, model, k: Optional[int] = None):
        import net
        return net.genome_eval_events(model, self.va_win, self.va_lab,
                                      self.va_fidx, self.device, k or self.k)

    def stream_recall(self, model, g):
        """Recall @ budżet FA/h na splicie val (pełne klipy, kind/group z files.csv).
        Klipy val ładowane raz i cache'owane. Zwraca (recall, report)."""
        import stream_eval_torch as st
        from stream_eval import primary_recall
        ch_in = g.layer_sizes()[0]
        if self._stream_clips is None:
            self._stream_clips = st.load_clip_spikes(self.val_dir, ch_in)
        rep = st.evaluate_stream(model, self.val_dir, ch_in, self.device,
                                 budgets=(self.stream_budget,),
                                 n_boot=self.stream_boot, clips=self._stream_clips)
        return primary_recall(rep, self.stream_budget), rep

    def stream_report_test(self, model, g, budgets=None, n_boot: int = 500):
        """Pełny raport strumieniowy na NIETKNIĘTYM splicie test (recall @ 1 i budżet
        FA/h, per kind, z CI po group_id). To jest liczba do nagłówka wyniku —
        nie ta z val, na której selekcjonowaliśmy."""
        import stream_eval_torch as st
        if not self.test_dir:
            raise ValueError("brak test_data — nie ma nietkniętego splitu do raportu")
        ch_in = g.layer_sizes()[0]
        if self._test_clips is None:
            self._test_clips = st.load_clip_spikes(self.test_dir, ch_in)
        budgets = budgets or (1.0, self.stream_budget)
        return st.evaluate_stream(model, self.test_dir, ch_in, self.device,
                                  budgets=tuple(budgets), n_boot=n_boot,
                                  clips=self._test_clips)

    def evaluate_on_test(self, g: Genome, epochs: Optional[int] = None,
                         model=None, n_boot: int = 500):
        """Wytrenuj genom (fresh) i policz jego raport na teście. Gdy `model` podany
        (np. z pełnego treningu zwycięzcy), używa go zamiast trenować od nowa."""
        if model is None:
            _, _, _, model = self.train_once(g, epochs or self.epochs, self.seed,
                                             return_model=True)
        return self.stream_report_test(model, g, n_boot=n_boot)

    def train_once(self, g: Genome, epochs: int, seed: int, return_model: bool = False):
        """Jeden trening (fresh init) + ocena. Zwraca (metrics, loss0, lossN[, model])."""
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
        metrics = self.eval_events(model)
        if return_model:
            return metrics, loss0, lossN, model
        return metrics, loss0, lossN

    def __call__(self, g: Genome, budget: float = 1.0) -> float:
        epochs = max(1, round(self.epochs * budget))
        need_model = self.metric == "recall_fa"
        aps, f1s, recs, l0, lN = [], [], [], 0.0, 0.0
        for si in range(self.fitness_seeds):
            out = self.train_once(g, epochs, self.seed + si, return_model=need_model)
            if need_model:
                m, loss0, lossN, model = out
                rec, _ = self.stream_recall(model, g)
                recs.append(rec)
            else:
                m, loss0, lossN = out
            aps.append(m["ap"]); f1s.append(m["clip_f1"])
            l0, lN = loss0, lossN
        ap = sum(aps) / len(aps)
        f1 = sum(f1s) / len(f1s)
        rec = sum(recs) / len(recs) if recs else 0.0
        n_feat = len(g.features_used())
        # kara parsymonii: drobny minus za okablowanie (fan-out) i za liczbę
        # użytych kanałów encodera — rozstrzyga remisy na korzyść MNIEJSZEGO
        # zestawu wejść (selekcja cech), nie przebijając realnych różnic jakości.
        base = {"ap": ap, "clip_f1": f1, "recall_fa": rec}[self.metric]
        score = base - self.fanout_penalty * _avg_fanout(g) \
            - self.feature_penalty * n_feat
        # GUARD anty-miraż: sieć, która przy progu k nic nie wykrywa (clipF1==0),
        # jest bezużyteczna — AP potrafi wtedy fałszywie wyjść 1.0 przy prawie
        # milczącym neuronie. Dla recall_fa recall==0 już jest podłogą.
        dead = f1 <= 1e-9 and self.metric != "recall_fa"
        if dead:
            score = 0.0
        if self.verbose:
            std = (sum((a - ap) ** 2 for a in aps) / len(aps)) ** 0.5
            extra = f"  recall@{self.stream_budget:g}FA/h {rec:.3f}" if need_model else ""
            print(f"    [eval] {g.layer_sizes()} loss {l0:.3f}->{lN:.3f}  "
                  f"AP {ap:.3f}±{std:.3f}  clipF1 {f1:.3f}{extra}  cechy {n_feat}"
                  f"{'  [MARTWY->0]' if dead else ''} (ep{epochs}, "
                  f"seedy{self.fitness_seeds})", flush=True)
        return score


def _load_channel_names(data_dir: str, n_channels: int):
    """Nazwy kanałów: sidecar channels.json obok danych (zbiory rozszerzone),
    inaczej pipeline CHANNELS (gdy pasuje liczebnością), inaczej ch{i}."""
    import json
    cands = [data_dir]
    if data_dir:
        cands.append(os.path.dirname(data_dir.rstrip("/\\")))
    for d in cands:
        p = os.path.join(d, "channels.json") if d else None
        if p and os.path.exists(p):
            try:
                names = json.load(open(p, encoding="utf-8"))
                if isinstance(names, dict):
                    names = names.get("channels", [])
                # przy --channels-head sidecar ma więcej nazw niż kanałów — obetnij
                if len(names) >= n_channels:
                    return list(names)[:n_channels]
            except Exception:
                pass
    try:
        from snn_hw_pipeline import CHANNELS as PIPE_CHANNELS
        if len(PIPE_CHANNELS) == n_channels:
            return list(PIPE_CHANNELS)
    except Exception:
        pass
    return [f"ch{i}" for i in range(n_channels)]


def _avg_fanout(g: Genome) -> float:
    counts = []
    for k in range(len(g.layers)):
        counts += [c for c in g.fanout_counts(k) if c > 0]
    return sum(counts) / max(len(counts), 1)
