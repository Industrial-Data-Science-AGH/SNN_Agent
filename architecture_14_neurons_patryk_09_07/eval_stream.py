#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_stream.py — symulacja pracy ciągłej: każdy klip przechodzi przez sieć W CAŁOŚCI
(bez cięcia na okna treningowe), a decyzję podejmuje reguła dekodera "k spików
neuronu D w oknie w ramek" — dokładnie to, co może zliczać Arduino na J4 płytki D.

To jest metryka, która przewiduje zachowanie na żywym demie: % wykrytych nagrań
szkła, % klipów tła z fałszywym alarmem i liczba alarmów na godzinę tła.

Użycie:
    python eval_stream.py --ckpt sweep_pw10_s1.pt --data spikes_manifest/val
    python eval_stream.py --ckpt best.pt --data spikes_manifest/test --rules 1:1 2:100 3:250
"""
from __future__ import annotations

import argparse, glob, os, sys
import numpy as np
import torch

from snn_hw_pipeline import LuiNet, DT, V_LEAK_MAX, _inv_map, CH_IN, infer_wide

# wspólny moduł metryki strumieniowej (count_alarms + recall@budżet FA/h) —
# jedno źródło logiki dla eval_stream / pipeline / GA
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from snn_pipeline.stream_eval import (  # noqa: E402
    count_alarms, stream_report, format_report, load_files_meta)

REFRAC_FRAMES = 500   # po alarmie system i tak budzi reaktor — 5 s martwej strefy


def load_clips(root):
    files = [f for f in sorted(glob.glob(os.path.join(root, "**", "*.csv"),
                                         recursive=True))
             if os.path.basename(f) != "files.csv"]   # files.csv to metadane, nie ramki
    if not files:
        sys.exit(f"brak CSV w {root}")
    clips = []
    for f in files:
        with open(f) as fh:
            header = fh.readline().strip().split(",")
            arr = np.loadtxt(fh, delimiter=",", ndmin=2, dtype=np.float32)
        if arr.size == 0:
            continue
        s = arr[:, 1:1 + CH_IN].astype(np.uint8)
        if "label" in header:
            y = int(arr[:, header.index("label")].max())
        else:
            nm = os.path.basename(f).lower()
            y = int("glass" in nm or "szklo" in nm)
        clips.append((s, y, f))
    return clips


@torch.no_grad()
def d_spike_trains(model, clips, bs=64):
    """Zwraca listę binarnych ciągów spików neuronu D, po jednym na klip.
    Klipy grupowane po długości i dopełniane zerami (zero = brak spiku wejścia,
    membrana tylko opada do V_leak — padding nie generuje fałszywych spików)."""
    order = np.argsort([len(c[0]) for c in clips])
    trains = [None] * len(clips)
    for lo in range(0, len(order), bs):
        idx = order[lo:lo + bs]
        T = max(len(clips[i][0]) for i in idx)
        x = torch.zeros(len(idx), T, CH_IN)
        for r, i in enumerate(idx):
            s = clips[i][0]
            x[r, :len(s)] = torch.from_numpy(s.astype(np.float32))
        so = model(x)["so"][..., 0].cpu().numpy()   # [B, T]
        for r, i in enumerate(idx):
            trains[i] = so[r, :len(clips[i][0])].astype(np.uint8)
    return trains


def clip_source(path):
    """Źródło klipu po wzorcu nazwy — do rozbicia fałszywych alarmów: ciche tło
    (notebooks/voice) to co innego niż trudne negatywy-zdarzenia (datasec/esc50)."""
    nm = os.path.basename(path)
    if "Background.output" in nm:
        return "notebooks-tło"
    if "synthetic_" in nm:
        return "voice"
    stem = nm.split("_", 2)[-1]
    if len(stem) > 1 and stem[0].isdigit() and "-" in stem[:12]:
        return "esc50"
    return "datasec"


def _clip_meta(data_dir, clips):
    """Dla każdego klipu dołóż (kind, group_id) z files.csv (fallback: clip_source)."""
    try:
        meta = load_files_meta(data_dir)
    except FileNotFoundError:
        meta = {}
    kinds, groups = [], []
    for _, y, path in clips:
        m = meta.get(os.path.basename(path))
        if m is not None:
            kinds.append(m.kind); groups.append(m.group_id)
        else:                                   # stary artefakt bez files.csv
            kinds.append("positive" if y == 1 else clip_source(path))
            groups.append(os.path.basename(path))
    return np.array(kinds, dtype=object), np.array(groups, dtype=object)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rules", nargs="*", default=["1:1", "2:100", "2:250", "3:250", "3:500"],
                    help="reguły k:w — k spików D w oknie w ramek (1 ramka = 10 ms)")
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--budgets", type=float, nargs="+", default=[1.0, 6.0],
                    help="budżety FA/h do raportu recall (domyślnie 1 i 6)")
    ap.add_argument("--boot", type=int, default=500,
                    help="liczba próbek bootstrap po group_id (0 = bez CI)")
    ap.add_argument("--d-leak-delta", type=float, default=0.0,
                    help="obniż V_leak neuronu D o tyle (twardszy próg decyzji; "
                         "fizycznie = niższy pasek LED na płytce D, zero zmian w treningu)")
    args = ap.parse_args()

    state = torch.load(args.ckpt, map_location="cpu")
    sd = state["model"] if "model" in state else state
    model = LuiNet(wide=infer_wide(sd))     # auto: 4 lub 8 płytek H
    model.load_state_dict(sd)
    if not args.no_quant:
        model.set_quantize(True)
    model.eval()

    if args.d_leak_delta:
        with torch.no_grad():
            vl = model.O.v_leak().item()
            target = min(max(vl - args.d_leak_delta, 0.001), V_LEAK_MAX - 0.001)
            model.O.p_vleak.data.fill_(_inv_map(target, 0.0, V_LEAK_MAX))
        print(f"[D] V_leak {vl:.3f} -> {target:.3f} (delta {args.d_leak_delta})")

    clips = load_clips(args.data)
    labels = np.array([c[1] for c in clips])
    print(f"[dane] {len(clips)} klipów: {int((labels==1).sum())} glass, "
          f"{int((labels==0).sum())} tła", flush=True)

    trains = d_spike_trains(model, clips)
    neg_hours = sum(len(t) for t, y in zip(trains, labels == 0) if y) * DT / 3600.0
    n_spk = np.array([t.sum() for t in trains])
    print(f"[D] średnio spików na klip: glass {n_spk[labels==1].mean():.2f}, "
          f"tło {n_spk[labels==0].mean():.2f}; tło łącznie {neg_hours:.2f} h")

    sources = np.array([clip_source(c[2]) for c in clips])
    src_names = sorted(set(sources[labels == 0]))

    hdr = f"\n{'reguła':>12} {'wykryte glass':>14} {'klipy tła FA':>13} {'alarmy/h tła':>13}"
    hdr += "".join(f" {('FA ' + s):>16}" for s in src_names)
    print(hdr)
    results = {}
    for rule in args.rules:
        k, w = (int(v) for v in rule.split(":"))
        det = np.array([count_alarms(t, k, w) > 0 for t in trains])
        fa_events = sum(count_alarms(t, k, w) for t, y in zip(trains, labels == 0) if y)
        rec = det[labels == 1].mean()
        fa = det[labels == 0].mean()
        fa_h = fa_events / max(neg_hours, 1e-9)
        results[rule] = (rec, fa, fa_h)
        line = f"  k={k} w={w*10:4d}ms {100*rec:13.1f}% {100*fa:12.1f}% {fa_h:13.1f}"
        for s in src_names:
            m = (labels == 0) & (sources == s)
            line += f" {100*det[m].mean():15.1f}%"
        print(line)

    # --- metryka DECYZYJNA: recall @ budżet FA/h, per kind, z CI (wspólny moduł) ---
    kinds, groups = _clip_meta(args.data, clips)
    n_frames = np.array([len(t) for t in trains], dtype=np.int64)
    report = stream_report(trains, labels, kinds, groups, n_frames=n_frames,
                           dt=DT, budgets=tuple(args.budgets), refrac=REFRAC_FRAMES,
                           n_boot=args.boot)
    print(f"\n[recall @ budżet FA/h  (per kind, CI bootstrap po group_id, "
          f"{len(set(groups))} grup)]")
    print(format_report(report))

    return results


if __name__ == "__main__":
    main()
