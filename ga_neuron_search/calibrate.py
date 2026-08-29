#!/usr/bin/env python3
"""
calibrate.py — dobór budżetu epok proxy-treningu (#2).

Problem: fitness po zbyt małej liczbie epok ma dużą wariancję (GA optymalizuje
szum), a po zbyt dużej — marnujemy budżet. Szukamy progu, gdzie loss jeszcze
WYRAŹNIE spada (sieć się uczy), a wariancja metryki między seedami jest już mała.

Ten skrypt bierze JEDNĄ losową topologię o zadanym N i trenuje ją dla kilku
wartości epok × kilku seedów, po czym drukuje tabelę:
  epoki | loss_end (śr) | AP śr±std | clipF1 śr±std
Wybierz najmniejsze `epoki`, gdzie std AP jest małe, a loss_end wyraźnie < loss(1 epoka).

Użycie:
    python calibrate.py --n 8 \
        --data ../architecture_14_neurons_patryk_09_07/spikes_manifest7/train \
        --val-data ../architecture_14_neurons_patryk_09_07/spikes_manifest7/val \
        --arch-dir ../architecture_14_neurons_patryk_09_07 \
        --epochs-grid 2 4 6 8 12 --seeds 3
"""

from __future__ import annotations

import argparse
import random

from fitness import RealFitness
from genome import random_genome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n", type=int, default=8, help="liczba neuronów topologii testowej"
    )
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--arch-dir", default="../architecture_14_neurons_patryk_09_07")
    ap.add_argument("--epochs-grid", type=int, nargs="+", default=[2, 4, 6, 8, 12])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--num-samples", type=int, default=6000)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--topo-seed", type=int, default=0)
    args = ap.parse_args()

    rf = RealFitness(
        arch_dir=args.arch_dir,
        data=args.data,
        val_data=args.val_data,
        epochs=max(args.epochs_grid),
        num_samples=args.num_samples,
        k=args.k,
        verbose=False,
    )
    g = random_genome(args.n, random.Random(args.topo_seed), max_hidden_layers=4)
    print(f"[calib] topologia testowa {g.layer_sizes()}  N={args.n}")
    print(g.pretty())

    def stat(xs):
        m = sum(xs) / len(xs)
        sd = (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5
        return m, sd

    print("\n" + "=" * 62)
    print(f"{'epoki':>6} | {'loss_end':>9} | {'AP sr±std':>16} | {'clipF1 sr±std':>16}")
    print("-" * 62)
    for ep in args.epochs_grid:
        aps, f1s, losses = [], [], []
        for si in range(args.seeds):
            m, loss0, lossN = rf.train_once(g, ep, seed=si)
            aps.append(m["ap"])
            f1s.append(m["clip_f1"])
            losses.append(lossN)
        ap_m, ap_s = stat(aps)
        f1_m, f1_s = stat(f1s)
        lo_m, _ = stat(losses)
        print(
            f"{ep:>6} | {lo_m:>9.3f} | {ap_m:>7.3f}±{ap_s:<7.3f} | "
            f"{f1_m:>7.3f}±{f1_s:<7.3f}"
        )
    print("=" * 62)
    print("Wybierz najmniejsze `epoki` z małym std AP i wyraznie nizszym loss_end.")


if __name__ == "__main__":
    main()
