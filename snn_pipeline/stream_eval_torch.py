#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stream_eval_torch.py — torchowa warstwa nad `stream_eval` (numpy).

Liczy per-klipowe ciągi spików neuronu decyzyjnego D DOWOLNYM modelem (LuiNet z
pipeline'u albo GenomeNet z GA — oba zwracają model(x)["so"] = [B,T,1]), dołącza
metadane `kind`/`group_id` z files.csv i zwraca raport recall @ budżet FA/h.

Wołane z:
  * snn_hw_pipeline.train  — selekcja checkpointu po recall@FA/h (zamiast F1),
  * ga_neuron_search/fitness — fitness GA po recall@FA/h,
  * eval_stream            — może korzystać z tej samej ścieżki.

Osobny plik (a nie w stream_eval), żeby stream_eval został czysto-numpyowy i
importowalny bez torcha (genome-side testy GA).
"""
from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# import odporny na uruchomienie jako moduł pakietu i jako luźny plik
try:
    from .stream_eval import (DEFAULT_BUDGETS, DEFAULT_DT, DEFAULT_REFRAC,
                              OperatingPoint, load_files_meta, stream_report)
except ImportError:  # uruchomione z dołożonym katalogiem snn_pipeline w sys.path
    from stream_eval import (DEFAULT_BUDGETS, DEFAULT_DT, DEFAULT_REFRAC,
                             OperatingPoint, load_files_meta, stream_report)


def load_clip_spikes(split_dir: str, ch_in: int):
    """Wczytaj pełne klipy (bez cięcia na okna) z katalogu splitu.

    Zwraca listę (spikes[n_frames, ch_in] uint8, label int, basename_csv).
    Pomija files.csv (metadane, nie ramki).
    """
    files = [f for f in sorted(glob.glob(os.path.join(split_dir, "**", "*.csv"),
                                         recursive=True))
             if os.path.basename(f) != "files.csv"]
    if not files:
        raise FileNotFoundError(f"brak CSV klipów w {split_dir}")
    clips = []
    for f in files:
        with open(f) as fh:
            header = fh.readline().strip().split(",")
            arr = np.loadtxt(fh, delimiter=",", ndmin=2, dtype=np.float32)
        if arr.size == 0:
            continue
        s = arr[:, 1:1 + ch_in].astype(np.uint8)
        if "label" in header:
            y = int(arr[:, header.index("label")].max())
        else:
            nm = os.path.basename(f).lower()
            y = int("glass" in nm or "szklo" in nm)
        clips.append((s, y, os.path.basename(f)))
    return clips


def d_spike_trains(model, clips, ch_in: int, device: str, bs: int = 64):
    """Binarne ciągi spików neuronu D, jeden na klip (grupowane po długości,
    dopełniane zerami — padding nie generuje spików, bo membrana tylko opada)."""
    import torch
    order = np.argsort([len(c[0]) for c in clips])
    trains: List[Optional[np.ndarray]] = [None] * len(clips)
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(order), bs):
            idx = order[lo:lo + bs]
            T = max(len(clips[i][0]) for i in idx)
            x = torch.zeros(len(idx), T, ch_in, device=device)
            for r, i in enumerate(idx):
                s = clips[i][0]
                x[r, :len(s)] = torch.from_numpy(s.astype(np.float32)).to(device)
            so = model(x)["so"][..., 0].cpu().numpy()   # [B, T]
            for r, i in enumerate(idx):
                trains[i] = so[r, :len(clips[i][0])].astype(np.uint8)
    return trains


def evaluate_stream(model, split_dir: str, ch_in: int, device: str,
                    budgets: Tuple[float, ...] = DEFAULT_BUDGETS,
                    dt: float = DEFAULT_DT, refrac: int = DEFAULT_REFRAC,
                    n_boot: int = 500, seed: int = 0, bs: int = 64,
                    clips=None) -> Dict[float, OperatingPoint]:
    """Pełna ocena strumieniowa modelu na splicie: recall @ budżet FA/h, per kind,
    z CI po group_id. `clips` można podać z zewnątrz (cache), inaczej ładuje sam."""
    if clips is None:
        clips = load_clip_spikes(split_dir, ch_in)
    trains = d_spike_trains(model, clips, ch_in, device, bs)

    try:
        meta = load_files_meta(split_dir)
    except FileNotFoundError:
        meta = {}
    labels, kinds, groups, n_frames = [], [], [], []
    for s, y, base in clips:
        m = meta.get(base)
        labels.append(y)
        n_frames.append(len(s))
        if m is not None:
            kinds.append(m.kind); groups.append(m.group_id)
        else:
            kinds.append("positive" if y == 1 else "background")
            groups.append(base)
    return stream_report(trains, np.array(labels), np.array(kinds, dtype=object),
                         np.array(groups, dtype=object),
                         n_frames=np.array(n_frames), dt=dt, budgets=budgets,
                         refrac=refrac, n_boot=n_boot, seed=seed)
