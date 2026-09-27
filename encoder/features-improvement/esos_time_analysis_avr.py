#!/usr/bin/env python3
"""
esos_time_analysis_avr.py — budżet czasu enkodera dla ATmega328P @16 MHz (Arduino Uno/Nano), bez FPU.

Zastępuje esos_time_analysis.py (Cortex-M4F @64 MHz, per ramka, 1 cykl/op). Tu:
  * koszt liczony PER PRÓBKĘ w ISR (to on ogranicza: musi się zmieścić w 1/fs), a nie per ramka;
  * przewidywane cykle pochodzą z ANALIZY ASEMBLERA prawdziwego firmware (predictions.json, tools/isr_cycles.py),
    zweryfikowanej symulatorem cyklowym simavr (błąd < 1%);
  * kolumny "PŁYTKA" czytane z measurements.json (wypełnij tools/capture_bench.py) — puste = "—".

Użycie:
    python3 esos_time_analysis_avr.py                                  # predictions.json + measurements.json
    python3 esos_time_analysis_avr.py --fs 19231 --budget-us 52
"""
from __future__ import annotations

import argparse
import json
import os

F_CPU = 16_000_000
FS_HZ_DESIGN = 19231          # fs zakładane w twinie/dokumentacji
HOP_SAMPLES = 192
BUDGET_US_CRITERION = 52.0    # kryterium akceptacji z zadania

# Cechy widmowe (FFT) z esos_time_analysis.py — NIE wyceniane dla tego MCU (patrz uwaga na końcu).
SPECTRAL = ["spectral_centroid", "dominant_freq", "band_energy_low", "band_energy_mid", "band_energy_high",
            "spectral_flatness", "spectral_flux"]

DESCR = {
    "baseline":   "oryginalny encoder_v2.ino",
    "acc32":      "baseline + akumulatory 32-bit (bez __adddi3)",
    "dcfix":      "baseline + poprawka DC (bez martwej strefy)",
    "parity":     "baseline + DC + HF round + EPS_FLOOR (zgodność z twinem)",
    "swap":       "wymiana kanałów (mobility, autocorr), akumulatory 64-bit",
    "swap_acc32": "wymiana kanałów + akumulatory 32-bit",
    "swap_full":  "wymiana kanałów + parity + akumulatory 32-bit  (DOCELOWY)",
}


def us(c, f_cpu=F_CPU):
    return 1e6 * c / f_cpu


def fmt(v, spec="{:.1f}"):
    return "—" if v is None else spec.format(v)


def load(path):
    if path and os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="predictions.json")
    ap.add_argument("--measurements", default="measurements.json")
    ap.add_argument("--fs", type=float, default=FS_HZ_DESIGN, help="fs do przeliczeń zajętości (domyślnie 19231)")
    ap.add_argument("--budget-us", type=float, default=BUDGET_US_CRITERION)
    ap.add_argument("--table-only", action="store_true", help="tylko tabela przewidywań")
    a = ap.parse_args()

    pred = load(a.predictions)
    if not pred:
        raise SystemExit(f"brak {a.predictions} — uruchom tools/run_predictions.sh")
    meas = load(a.measurements)
    board = meas.get("board", {})
    period_us = 1e6 / a.fs
    base = pred.get("baseline")

    print(f"=== ATmega328P @ {F_CPU // 1_000_000} MHz, fs = {a.fs:g} Hz (okres próbki {period_us:.2f} us; kryterium ISR <= {a.budget_us:g} us) ===")
    print("PRZEWIDYWANIE = statyczna analiza asemblera (min/max ścieżki, z wejściem w przerwanie 7 cykli).\n")
    hdr = f"{'wariant':11s} {'cykle min':>9s} {'cykle max':>9s} {'us max':>7s} {'% okresu':>8s} {'Δ vs base':>9s} {'push':>4s} {'flash':>6s} {'RAM':>4s}  kryterium"
    print(hdr)
    print("-" * len(hdr))
    for name, p in pred.items():
        d = p["isr_cycles_max"] - base["isr_cycles_max"] if base else 0
        u = us(p["isr_cycles_max"])
        ok = "OK" if u <= a.budget_us else "PRZEKROCZONE"
        print(f"{name:11s} {p['isr_cycles_min']:9d} {p['isr_cycles_max']:9d} {u:7.2f} {100*u/period_us:7.1f}% {d:+9d} "
              f"{p['pushes']:4d} {p['flash_bytes']:6d} {p['ram_bytes']:4d}  {ok}  — {DESCR.get(name, '')}")
    if a.table_only:
        return

    print("\n=== PŁYTKA (measurements.json) vs PRZEWIDYWANIE ===")
    h2 = f"{'wariant':11s} {'fs [Hz]':>9s} {'ISR CPU%':>8s} {'ISR us śr.':>10s} {'cykle śr.':>9s} {'przew. [min,max]':>17s} {'w zakresie ±3%':>15s}"
    print(h2)
    print("-" * len(h2))
    for name, p in pred.items():
        b = board.get(name, {})
        cyc = b.get("isr_cycles_mean")
        inr = "—"
        if cyc is not None:
            inr = "TAK" if p["isr_cycles_min"] * 0.97 <= cyc <= p["isr_cycles_max"] * 1.03 else "NIE — model do poprawy"
        print(f"{name:11s} {fmt(b.get('fs_hz'), '{:.0f}'):>9s} {fmt(b.get('isr_cpu_pct')):>8s} {fmt(b.get('isr_us_mean'), '{:.2f}'):>10s} "
              f"{fmt(cyc):>9s} {str(p['isr_cycles_min']) + ',' + str(p['isr_cycles_max']):>17s} {inr:>15s}")

    print("\n=== BUDŻET loop() (przetwarzanie ramki, float bez FPU + Serial) — POMIAR z płytki ===")
    h3 = f"{'wariant':11s} {'faza':>5s} {'ramek':>6s} {'spóźn.':>7s} {'s_n max':>7s} {'proc śr. [ms]':>13s} {'proc max [ms]':>13s} {'okres śr. [ms]':>14s}  werdykt"
    print(h3)
    print("-" * len(h3))
    for name in pred:
        b = board.get(name, {})
        for ph, lab in (("p1", "druk"), ("p2", "cicho")):
            s = b.get(ph)
            if not s:
                print(f"{name:11s} {lab:>5s} {'—':>6s} {'—':>7s} {'—':>7s} {'—':>13s} {'—':>13s} {'—':>14s}")
                continue
            per_ms = s["per_us_mean"] / 1000 if s.get("per_us_mean") else None
            ver = "OK" if s["proc_us_max"] < 0.8 * (per_ms or 10) * 1000 and s["late"] == 0 else \
                  ("UWAGA: ramki spóźnione (dryf siatki 192)" if s["late"] else "UWAGA: proc > 80% okresu")
            print(f"{name:11s} {lab:>5s} {s['frames']:6d} {s['late']:7d} {s['sn_max']:7d} {s['proc_us_mean']/1000:13.2f} {s['proc_us_max']/1000:13.2f} "
                  f"{fmt(per_ms, '{:.3f}'):>14s}  {ver}")

    sc = [(n, board.get(n, {}).get("scope")) for n in pred]
    if any(x and x.get("isr_pulse_us_max") for _, x in sc):
        print("\n=== OSCYLOSKOP (opcjonalnie, ENC_ISR_PIN=1; puls NIE obejmuje prologu/epilogu ISR ≈ 2*2*liczba_push cykli) ===")
        for n, x in sc:
            if x and x.get("isr_pulse_us_max"):
                extra = 4 * pred[n]["pushes"] / F_CPU * 1e6
                print(f"  {n:11s}: puls min {fmt(x.get('isr_pulse_us_min'), '{:.2f}')} us, max {fmt(x.get('isr_pulse_us_max'), '{:.2f}')} us"
                      f"  +prolog/epilog ~{extra:.1f} us  => ISR max ~{x['isr_pulse_us_max'] + extra:.1f} us")

    fs_b = [b.get("fs_hz") for b in board.values() if b.get("fs_hz")]
    if fs_b and abs(fs_b[0] - FS_HZ_DESIGN) / FS_HZ_DESIGN > 0.05:
        print(f"\n!!! UWAGA: zmierzone fs = {fs_b[0]:.0f} Hz ≠ {FS_HZ_DESIGN} Hz założone w twinie (HOP={HOP_SAMPLES} próbek = "
              f"{1e3*HOP_SAMPLES/fs_b[0]:.2f} ms, nie 10 ms). Sprawdź prescaler ADC (ENC_ADC_PRESCALER=64) i budżet: {1e6/fs_b[0]:.1f} us.")

    print("\nCechy widmowe (FFT) z esos_time_analysis.py — " + ", ".join(SPECTRAL) + ":")
    print("  NIE wyceniane dla ATmega328P (brak FPU, 2 KB RAM; tabela M4F '1.7% CPU' tu nie obowiązuje). "
          "Każdy wniosek 'dodajmy cechy widmowe' wymaga osobnego pomiaru na tym MCU.")


if __name__ == "__main__":
    main()
