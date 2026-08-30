#!/usr/bin/env python3
"""Tabela wyniku odniesienia z plików models/hw_<wersja>_s*.json.

Powód istnienia: dotychczasowe tabele w PRZEWODNIK_KOMPLETNY.md, kalibracja_sciaga_v3.md
i WYNIKI.md były PRZEPISYWANE RĘCZNIE z jednego wybranego seeda, bez zapisanego
pochodzenia. Nie dało się sprawdzić, na czym model się uczył ani czy liczby są
z tego samego artefaktu. Ten skrypt czyta pochodzenie z każdego configu, sprawdza,
że wszystkie seedy pochodzą z tego samego artefaktu, i wypisuje markdown.

Raportujemy ŚREDNIĄ I ROZRZUT po seedach, nie maksimum. Zmierzony rozrzut val-F1
między seedami to ~0.06, czyli więcej niż różnice, o które toczą się spory.

    python report_baseline.py --models models --glob "hw_v2_s*.json" > models/WYNIKI_v2.md
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st


def _fmt(vals, prec=3):
    """mediana i rozrzut min-max po seedach."""
    if not vals:
        return "n/a"
    if len(vals) == 1:
        return f"{vals[0]:.{prec}f}"
    return f"{st.median(vals):.{prec}f} ({min(vals):.{prec}f}–{max(vals):.{prec}f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models")
    ap.add_argument("--glob", default="hw_v2_s*.json")
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.models, a.glob)))
    if not paths:
        raise SystemExit(f"brak plików {a.glob} w {a.models}")

    cfgs = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            c = json.load(fh)
        c["_path"] = p
        cfgs.append(c)

    provs = {json.dumps(c.get("model_provenance", {}).get("train_data", {}), sort_keys=True)
             for c in cfgs}
    if len(provs) > 1:
        raise SystemExit("seedy pochodzą z RÓŻNYCH artefaktów treningowych — "
                         "nie wolno ich uśredniać; sprawdź model_provenance.train_data")
    prov = cfgs[0].get("model_provenance", {})
    td = prov.get("train_data", {})

    print(f"# Wynik odniesienia — {td.get('dataset_version', '?')}, podział grupowy\n")
    print("Wygenerowane przez `report_baseline.py`, nie przepisywane ręcznie.\n")
    print("## Na czym to policzono\n")
    print(f"- artefakt spike'owy: `{td.get('artifact_dir', '?')}`")
    print(f"- wersja zbioru: **{td.get('dataset_version', '?')}**, "
          f"manifest `{str(td.get('manifest_sha256'))[:12]}`, "
          f"enkoder `{str(td.get('encoder_sha256'))[:12]}`")
    print(f"- podział: **grupowy** (`group_id` rozłączne między splitami, "
          f"asercja w `encoder_twin.build_manifest`)")
    print(f"- kolejność strumienia: {td.get('stream_order', '?')}")
    print(f"- konfiguracja: pos_weight {prov.get('pos_weight')}, "
          f"hat_frac {prov.get('hat_frac')}, epok {prov.get('epochs')}, "
          f"kwantyzacja {prov.get('quantized')}")
    print(f"- seedy: {', '.join(str(c['model_provenance']['seed']) for c in cfgs)} "
          f"({len(cfgs)} przebiegów)")
    print(f"- commit: `{prov.get('git_commit')}`\n")

    def col(sciezka, klucz):
        out = []
        for c in cfgs:
            d = c.get(sciezka)
            if isinstance(d, dict) and klucz in d:
                out.append(float(d[klucz]))
        return out

    print("## Metryki (mediana po seedach, w nawiasie min–max)\n")
    print("| metryka | walidacja | test |")
    print("|---|---|---|")
    for etykieta, klucz in (("recall (okna)", "recall"),
                            ("precyzja (okna)", "precision"),
                            ("F1 (okna)", "f1")):
        print(f"| {etykieta} | {_fmt(col('val_metrics', klucz))} "
              f"| {_fmt(col('test_metrics', klucz))} |")
    rob = [c["robustness"]["f1_mean"] for c in cfgs if "robustness" in c]
    robmin = [c["robustness"]["f1_min"] for c in cfgs if "robustness" in c]
    print(f"| F1 pod rozrzutem sprzętu (mean) | {_fmt(rob)} | — |")
    print(f"| F1 pod rozrzutem sprzętu (min) | {_fmt(robmin)} | — |")

    print("\n## Poziom klipów\n")
    ev = [c.get("test_events") for c in cfgs if c.get("test_events")]
    if ev and isinstance(ev[0], dict):
        klucze = sorted({k for e in ev for k in e if isinstance(e.get(k), (int, float))})
        print("| metryka klipowa | test |")
        print("|---|---|")
        for k in klucze:
            print(f"| {k} | {_fmt([float(e[k]) for e in ev if k in e])} |")
    else:
        print("_brak `test_events` w configach_")

    print("\n## Per seed\n")
    print("| seed | val F1 | test F1 | rob F1 min | checkpoint |")
    print("|---|---|---|---|---|")
    for c in cfgs:
        mp = c.get("model_provenance", {})
        print(f"| {mp.get('seed')} "
              f"| {c.get('val_metrics', {}).get('f1', float('nan')):.3f} "
              f"| {c.get('test_metrics', {}).get('f1', float('nan')):.3f} "
              f"| {c.get('robustness', {}).get('f1_min', float('nan')):.3f} "
              f"| `{mp.get('checkpoint', '?')}` |")

    print("\n## Jak czytać\n")
    print("Te liczby NIE są porównywalne z `hw7_config.json` ani z tabelami "
          "w `PRZEWODNIK_KOMPLETNY.md` §15 i `kalibracja_sciaga_v3.md`. Tamte powstały "
          "na artefakcie `spikes_manifest7`, w którym 194 z 194 miksów VOICe obecnych "
          "w teście było też w treningu, a klasa pozytywna ESC-50 to był `clock_tick`. "
          "Spadek względem tamtych liczb jest oczekiwany i jest miarą tego, ile "
          "poprzedni wynik brał z przecieku.")


if __name__ == "__main__":
    main()
