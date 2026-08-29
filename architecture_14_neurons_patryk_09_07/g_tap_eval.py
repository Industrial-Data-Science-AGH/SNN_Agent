#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
g_tap_eval.py — plan B dekodera: zamiast patrzeć tylko na spik płytki D, Arduino
podsłuchuje też linie G0/G1/G2 (wejścia D; odczyt równoległy jest elektrycznie
neutralny — wejście MCU to wysoka impedancja) i decyduje regułą liniową na
zliczeniach spików w przesuwnym oknie:

    alarm gdy  a0*n(G0) + a1*n(G1) + a2*n(G2) >= θ   w oknie w ramek

To jest jedna linijka w firmware dekodera. Skrypt przeszukuje wszystkie małe
całkowite kombinacje (a_i ∈ {-2..2}, θ ∈ {1..4}, w ∈ {50,100,250}) na walidacji
i wypisuje front Pareto (wykrywalność szkła vs fałszywe alarmy).

Użycie:
    python g_tap_eval.py --ckpt sweep_pw10_s1.pt --data spikes_manifest/val
"""
from __future__ import annotations

import argparse
import itertools
import numpy as np
import torch

from snn_hw_pipeline import LuiNet, DT, CH_IN
from eval_stream import load_clips, clip_source


@torch.no_grad()
def g_spike_trains(model, clips, bs=64):
    order = np.argsort([len(c[0]) for c in clips])
    trains = [None] * len(clips)
    for lo in range(0, len(order), bs):
        idx = order[lo:lo + bs]
        T = max(len(clips[i][0]) for i in idx)
        x = torch.zeros(len(idx), T, CH_IN)
        for r, i in enumerate(idx):
            s = clips[i][0]
            x[r, :len(s)] = torch.from_numpy(s.astype(np.float32))
        sg = model(x)["sg"].cpu().numpy()          # [B, T, 3]
        for r, i in enumerate(idx):
            trains[i] = sg[r, :len(clips[i][0])].astype(np.int16)
    return trains


def windowed_counts(train, w):
    """n_i[t] = liczba spików neuronu i w ostatnich w ramkach."""
    c = np.cumsum(np.vstack([np.zeros((1, train.shape[1]), dtype=np.int32), train]), axis=0)
    lo = np.maximum(0, np.arange(1, len(train) + 1) - w)
    return c[1:] - c[lo]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--windows", type=int, nargs="*", default=[50, 100, 250])
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    model = LuiNet()
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state["model"] if "model" in state else state)
    model.set_quantize(True)
    model.eval()

    clips = load_clips(args.data)
    labels = np.array([c[1] for c in clips])
    sources = np.array([clip_source(c[2]) for c in clips])
    quiet = (labels == 0) & (sources == "notebooks-tło")
    neg_hours = sum(len(c[0]) for c, y in zip(clips, labels == 0) if y) * DT / 3600.0
    print(f"[dane] {len(clips)} klipów ({int((labels==1).sum())} glass), "
          f"tło {neg_hours:.2f} h", flush=True)

    trains = g_spike_trains(model, clips)
    rates = np.stack([t.mean(axis=0) / DT for t in trains])
    for i in range(3):
        print(f"[G{i}] średnia częstość: glass {rates[labels==1, i].mean():.2f} Hz, "
              f"tło {rates[labels==0, i].mean():.2f} Hz")

    results = []
    for w in args.windows:
        counts = [windowed_counts(t, w) for t in trains]      # lista [T,3]
        for a in itertools.product((-2, -1, 0, 1, 2), repeat=3):
            if all(x <= 0 for x in a):
                continue
            combo_max = np.array([(c @ np.array(a)).max() if len(c) else -99
                                  for c in counts])
            for th in (1, 2, 3, 4):
                det = combo_max >= th
                rec = det[labels == 1].mean()
                fa = det[labels == 0].mean()
                fa_q = det[quiet].mean() if quiet.any() else 0.0
                results.append((rec, fa, fa_q, a, th, w))

    # front Pareto: maksymalny recall przy danym (zaokrąglonym) FA
    results.sort(key=lambda r: (round(r[1], 3), -r[0]))
    pareto, best_rec_at = [], -1.0
    for r in sorted(results, key=lambda r: r[1]):
        if r[0] > best_rec_at:
            pareto.append(r)
            best_rec_at = r[0]

    print(f"\n{'reguła (a0,a1,a2)>=θ, okno':>34} {'glass':>7} {'FA tła':>7} {'FA cisza':>9}")
    shown = 0
    for rec, fa, fa_q, a, th, w in pareto:
        if rec < 0.30:
            continue
        print(f"  {str(a):>16} >= {th}, {w*10:4d}ms {100*rec:6.1f}% {100*fa:6.1f}% "
              f"{100*fa_q:8.1f}%")
        shown += 1
        if shown >= args.top:
            break


if __name__ == "__main__":
    main()
