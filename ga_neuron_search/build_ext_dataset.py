#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_ext_dataset.py — buduje ROZSZERZONY zbiór spike'ów (14 kanałów) dla GA,
żeby GA mogło WYBIERAĆ spośród mocniejszych cech encodera, nie tylko 7 HW.

DLACZEGO OSOBNY ZBIÓR (uczciwie):
  Źródłowego audio dla spikes_manifest7 NIE ma w repo (0% plików CSV mapuje się
  na dostępne .wav). Jedyne surowe audio to voice_extracted/{glass,hard_negative}
  (VOICe "synthetic"). Ten skrypt buduje z niego samodzielny zbiór — do ćwiczenia
  SELEKCJI CECH przez GA, NIE do porównań 1:1 z oryginalną siecią (inny dataset,
  hard_negative to zdarzenia a nie stacjonarne tło).

KANAŁY (14): 7 HW z encoder_twin (identyczne) + 7 z feature_bank:
  HW : peak, peak_cnt, cv, zcr, flux, hf_lo, hf_hi        (encoder_twin.encode_file)
  NEW: hjorth_mobility, autocorr_lag1, curve_length, crest, spectral_flatness,
       spectral_centroid, band_energy_low
  Nowe kanały to cechy POZIOMU/KSZTAŁTU -> kodowane progiem BEZWZGLĘDNYM
  (jak hf w encoder_twin), autokalibrowanym do zadanego odsetka spików na TLE
  (--bg-rate). Kierunek (v>thr vs v<thr) wybierany po stronie, na której skupiają
  się pozytywy (Cohen's d). Refrakcja 1 ramka, jak w HW.

Cechy liczone na TYM SAMYM sygnale i ramkowaniu co encoder_twin (x=_remove_dc(
wav_to_adc_codes), HOP_SAMPLES), więc 7 HW jest 1:1, a nowe są frame-aligned.

Wyjście: <out>/{train,val,test}/_cache_T{T}_s{stride}.npz (format CachedClips)
         + <out>/{split}/channels.json (nazwy 14 kanałów).

Użycie (z katalogu ga_neuron_search):
  ..\\SNN\\Scripts\\python.exe build_ext_dataset.py --out spikes_ext
  # szybka próba: --limit 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# nazwy kanałów -----------------------------------------------------------------
HW_NAMES = ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_lo", "hf_hi"]
NEW_NAMES = ["hjorth_mobility", "autocorr_lag1", "curve_length", "crest",
             "spectral_flatness", "spectral_centroid", "band_energy_low"]
ALL_NAMES = HW_NAMES + NEW_NAMES
REFRAC_FRAMES = 1
EPS = 1e-6


def _import_encoder(arch_dir: str):
    arch_dir = os.path.abspath(arch_dir)
    if arch_dir not in sys.path:
        sys.path.insert(0, arch_dir)
    import encoder_twin as et
    return et


def compute_new_features(x_f: np.ndarray, fs: float) -> np.ndarray:
    """[n_frames, HOP] sygnału -> [n_frames, 7] cech ciągłych (kolejność NEW_NAMES)."""
    n = x_f.shape[1]
    ax = np.abs(x_f)
    peak = ax.max(axis=1)
    rms = np.sqrt((x_f ** 2).mean(axis=1))
    crest = peak / (rms + EPS)

    dx = np.diff(x_f, axis=1)
    var_x = x_f.var(axis=1)
    var_dx = dx.var(axis=1)
    hjorth = np.sqrt(var_dx / (var_x + EPS))

    num_ac = (x_f[:, :-1] * x_f[:, 1:]).sum(axis=1)
    den_ac = (x_f ** 2).sum(axis=1) + EPS
    autocorr = num_ac / den_ac

    curve = np.abs(dx).sum(axis=1) / n

    win = np.hanning(n)
    mag = np.abs(np.fft.rfft(x_f * win, axis=1))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    msum = mag.sum(axis=1) + EPS
    centroid = (freqs[None, :] * mag).sum(axis=1) / msum
    flat = np.exp(np.log(mag + EPS).mean(axis=1)) / (mag.mean(axis=1) + EPS)
    low = freqs < (fs / 2.0) * 0.15
    band_low = mag[:, low].sum(axis=1) / msum

    return np.column_stack([hjorth, autocorr, curve, crest,
                            flat, centroid, band_low]).astype(np.float32)


def _frame_signal(et, path: str) -> np.ndarray:
    """Ten sam sygnał co w encoder_twin.encode_file, poramkowany [n_frames, HOP]."""
    codes = et.wav_to_adc_codes(path)
    x = et._remove_dc(codes)
    nf = len(x) // et.HOP_SAMPLES
    if nf == 0:
        return np.zeros((0, et.HOP_SAMPLES), dtype=np.float64)
    return x[: nf * et.HOP_SAMPLES].reshape(nf, et.HOP_SAMPLES)


def _apply_threshold(vals: np.ndarray, thr: float, direction: int) -> np.ndarray:
    """Kod progowy + refrakcja (jak kanały hf w encoder_twin)."""
    fire = (vals > thr) if direction > 0 else (vals < thr)
    out = np.zeros(len(vals), dtype=np.uint8)
    refr = 0
    for k in range(len(vals)):
        if fire[k] and refr == 0:
            out[k] = 1
            refr = REFRAC_FRAMES
        elif refr > 0:
            refr -= 1
    return out


def _split_files(files, val_frac, test_frac, rng):
    idx = list(range(len(files)))
    rng.shuffle(idx)
    n = len(files)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    test = {files[i] for i in idx[:n_test]}
    val = {files[i] for i in idx[n_test:n_test + n_val]}
    return val, test


def _window(spikes, y, fi, T, stride, wins, labs, fidxs):
    s = spikes
    if len(s) < T:
        s = np.pad(s, ((0, T - len(s)), (0, 0)))
    for i in range(0, len(s) - T + 1, stride):
        wins.append(s[i:i + T]); labs.append(y); fidxs.append(fi)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arch-dir", default="../architecture_14_neurons_patryk_09_07")
    ap.add_argument("--glass-dir", default=None, help="domyślnie <arch>/voice_extracted/glass")
    ap.add_argument("--neg-dir", default=None, help="domyślnie <arch>/voice_extracted/hard_negative")
    ap.add_argument("--out", default="spikes_ext")
    ap.add_argument("--T", type=int, default=200)
    ap.add_argument("--stride", type=int, default=50)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--warmup-seconds", type=float, default=30.0)
    ap.add_argument("--bg-rate", type=float, default=0.08,
                    help="docelowy odsetek ramek ze spikiem na TLE dla nowych kanałów")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None, help="max plików/klasę (smoke test)")
    args = ap.parse_args()

    et = _import_encoder(args.arch_dir)
    import glob
    glass_dir = args.glass_dir or os.path.join(args.arch_dir, "voice_extracted", "glass")
    neg_dir = args.neg_dir or os.path.join(args.arch_dir, "voice_extracted", "hard_negative")
    glass = sorted(glob.glob(os.path.join(glass_dir, "**", "*.wav"), recursive=True))
    neg = sorted(glob.glob(os.path.join(neg_dir, "**", "*.wav"), recursive=True))
    if args.limit:
        glass, neg = glass[:args.limit], neg[:args.limit]
    if not glass or not neg:
        sys.exit(f"brak .wav (glass={len(glass)} w {glass_dir}, neg={len(neg)} w {neg_dir})")
    print(f"[dane] glass {len(glass)} (poz.), hard_negative {len(neg)} (neg.)  "
          f"kanały={len(ALL_NAMES)}: {','.join(ALL_NAMES)}", flush=True)

    rng = __import__("random").Random(args.seed)
    g_val, g_test = _split_files(glass, args.val_frac, args.test_frac, rng)
    n_val, n_test = _split_files(neg, args.val_frac, args.test_frac, rng)

    def split_of(f, val_set, test_set):
        return "test" if f in test_set else ("val" if f in val_set else "train")

    # warmup jednego ciągłego stanu na negatywach TRENINGOWYCH
    neg_train = [f for f in neg if split_of(f, n_val, n_test) == "train"]
    state, n_used = et._warmup_state(neg_train, args.warmup_seconds)
    used = set(neg_train[:n_used])

    # strumień: najpierw pozostałe negatywy, potem glass (jak ciągła praca urządzenia)
    stream = [(0, f) for f in neg if f not in used] + [(1, f) for f in glass]

    # PASS: policz 7 HW (encode_file) + 7 nowych (ciągłe) frame-aligned
    recs = []  # (label, split, hw[n,7], new[n,7])
    t0 = time.time()
    for i, (label, f) in enumerate(stream):
        hw = et.encode_file(f, state=state)          # [n,7] uint8, mutuje state
        if hw.shape[0] == 0:
            continue
        x_f = _frame_signal(et, f)
        new = compute_new_features(x_f, et.FS_HZ)    # [nf,7]
        m = min(len(hw), len(new))
        if m == 0:
            continue
        vset, tset = (g_val, g_test) if label == 1 else (n_val, n_test)
        recs.append((label, split_of(f, vset, tset), hw[:m], new[:m]))
        if (i + 1) % 500 == 0:
            print(f"[enc] {i+1}/{len(stream)} plików ({time.time()-t0:.0f}s)", flush=True)
    print(f"[enc] gotowe: {len(recs)} plików w {time.time()-t0:.0f}s", flush=True)

    # KALIBRACJA progów nowych kanałów na ramkach TRENINGOWYCH (thr z NEGATYWÓW)
    tr_new_pos = np.concatenate([r[3] for r in recs if r[1] == "train" and r[0] == 1]) \
        if any(r[1] == "train" and r[0] == 1 for r in recs) else np.zeros((0, len(NEW_NAMES)), np.float32)
    tr_new_neg = np.concatenate([r[3] for r in recs if r[1] == "train" and r[0] == 0]) \
        if any(r[1] == "train" and r[0] == 0 for r in recs) else np.zeros((0, len(NEW_NAMES)), np.float32)
    if len(tr_new_neg) == 0:
        sys.exit("brak ramek treningowych tła do kalibracji progów")

    dirs, thrs = [], []
    print("\n[kalibracja nowych kanałów]  (thr z tła, kierunek z separacji)")
    for c, name in enumerate(NEW_NAMES):
        neg_vals = tr_new_neg[:, c]
        pos_mean = tr_new_pos[:, c].mean() if len(tr_new_pos) else neg_vals.mean()
        direction = 1 if pos_mean >= neg_vals.mean() else -1
        q = (1.0 - args.bg_rate) if direction > 0 else args.bg_rate
        thr = float(np.quantile(neg_vals, q))
        dirs.append(direction); thrs.append(thr)
        cmp = ">" if direction > 0 else "<"
        print(f"  {name:20s} fire gdy v {cmp} {thr:.4f}  "
              f"(neg_śr {neg_vals.mean():.4f}, pos_śr {pos_mean:.4f})")

    # PROGOWANIE nowych -> spiki, sklej [HW | NEW], zbierz per split
    out = Path(args.out)
    split_data = {s: ([], [], [], []) for s in ("train", "val", "test")}  # wins,labs,fidx,filelab
    fire_stat = {0: np.zeros(len(ALL_NAMES)), 1: np.zeros(len(ALL_NAMES))}
    fire_frames = {0: 0, 1: 0}
    for label, split, hw, new in recs:
        new_sp = np.zeros_like(new, dtype=np.uint8)
        for c in range(len(NEW_NAMES)):
            new_sp[:, c] = _apply_threshold(new[:, c], thrs[c], dirs[c])
        spikes = np.concatenate([hw.astype(np.uint8), new_sp], axis=1)  # [n,14]
        wins, labs, fidxs, filelab = split_data[split]
        fi = len(filelab)
        filelab.append(float(label))
        _window(spikes, float(label), fi, args.T, args.stride, wins, labs, fidxs)
        fire_stat[label] += spikes.sum(axis=0)
        fire_frames[label] += len(spikes)

    # ZAPIS per split
    for split, (wins, labs, fidxs, filelab) in split_data.items():
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        if not wins:
            print(f"[!] {split}: 0 okien — pomijam"); continue
        win = np.stack(wins).astype(np.uint8)
        lab = np.array(labs, dtype=np.float32)
        fidx = np.array(fidxs, dtype=np.int32)
        file_lab = np.array(filelab, dtype=np.float32)
        sig = f"ext14|{len(filelab)}|{args.T}|{args.stride}|{args.bg_rate}"
        cache = d / f"_cache_T{args.T}_s{args.stride}.npz"
        np.savez_compressed(cache, win=win, lab=lab, fidx=fidx,
                            file_lab=file_lab, sig=sig)
        json.dump({"channels": ALL_NAMES, "n_hw": len(HW_NAMES),
                   "thresholds": thrs, "directions": dirs, "bg_rate": args.bg_rate},
                  open(d / "channels.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"[ok] {split}: {len(lab)} okien / {len(filelab)} klipów, "
              f"pozytywnych {int(lab.sum())} ({100*lab.mean():.1f}%) -> {cache}")

    # RAPORT: firing-rate per kanał per klasa (podgląd separacji)
    print(f"\n[firing-rate %]  ({'kanal':20s}   glass |  tlo  | roznica)")
    for c, name in enumerate(ALL_NAMES):
        rp = 100 * fire_stat[1][c] / max(fire_frames[1], 1)
        rn = 100 * fire_stat[0][c] / max(fire_frames[0], 1)
        flag = "  <-- separuje" if abs(rp - rn) >= 8 else ""
        print(f"  {name:20s} {rp:6.1f} | {rn:5.1f} | {rp-rn:+6.1f}{flag}")

    print(f"\n[gotowe] zbiór 14-kanałowy w {out}/  "
          f"(train/val/test + channels.json). Użyj w GA:\n"
          f"  --data {out}/train --val-data {out}/val")


if __name__ == "__main__":
    main()
