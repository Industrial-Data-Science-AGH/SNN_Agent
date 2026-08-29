#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
voice_extract.py — wycina klipy zdarzeń z długich, oznaczonych czasowo nagrań
datasetu VOICe (dataset/clean/annotation + dataset/clean/audio), żeby zasilić
build_dataset() z encoder_twin.py dodatkowymi przykładami treningowymi.

Format adnotacji (dataset/clean/annotation/synthetic_NNN.txt):
    start_s<TAB>end_s<TAB>label      label in {glassbreak, gunshot, babycry}

Dlaczego gunshot/babycry jako "trudne negatywy":
    Poprzedni trening (57 klipów glass z freesound vs 949 klipów tła) osiągnął
    tylko 28% precyzji — kanały peak/crest reagują na DOWOLNY ostry transient,
    nie tylko szkło. gunshot i babycry to realne, ostre zdarzenia dźwiękowe,
    które NIE są szkłem — trenowanie na nich jako negatywach zmusza sieć do
    uczenia się cech specyficznych dla szkła (crest/cv/peak_cnt), zamiast
    "dowolny głośny dźwięk = alarm".

Segmenty wycinane są z paddingiem (domyślnie 0.3 s z każdej strony), przycięte
do granic pliku. Zapisywane jako osobne .wav (ten sam sample rate co źródło) —
encoder_twin.wav_to_adc_codes i tak resampluje do FS_HZ przy kodowaniu.

Użycie:
    python voice_extract.py --annotation ../dataset/clean/annotation \
        --audio ../dataset/clean/audio --out ./voice_extracted
"""
from __future__ import annotations

import argparse
import glob
import os
import random
from pathlib import Path
from typing import List, Tuple

import soundfile as sf


def _parse_annotation(path: str) -> List[Tuple[float, float, str]]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            start, end, label = parts
            events.append((float(start), float(end), label))
    return events


def _extract_segment(y, sr: int, start: float, end: float, pad: float) -> "any":
    lo = max(0, int((start - pad) * sr))
    hi = min(len(y), int((end + pad) * sr))
    return y[lo:hi]


def extract(annotation_dir: str, audio_dir: str, out_dir: str,
            pad: float = 0.3, hard_negative_labels=("gunshot", "babycry"),
            hard_negative_sample: int | None = None, seed: int = 0) -> None:
    out = Path(out_dir)
    glass_out = out / "glass"
    hardneg_out = out / "hard_negative"
    glass_out.mkdir(parents=True, exist_ok=True)
    hardneg_out.mkdir(parents=True, exist_ok=True)

    ann_files = sorted(glob.glob(os.path.join(annotation_dir, "*.txt")))
    if not ann_files:
        print(f"[!] brak adnotacji w {annotation_dir}")
        return

    glass_events: List[Tuple[str, float, float]] = []
    hardneg_events: List[Tuple[str, float, float]] = []

    for ann_path in ann_files:
        stem = Path(ann_path).stem  # np. synthetic_001
        audio_path = os.path.join(audio_dir, f"{stem}.wav")
        if not os.path.exists(audio_path):
            print(f"[!] brak audio dla {ann_path} (szukano {audio_path}) — pomijam")
            continue
        for start, end, label in _parse_annotation(ann_path):
            if label == "glassbreak":
                glass_events.append((audio_path, start, end))
            elif label in hard_negative_labels:
                hardneg_events.append((audio_path, start, end))

    print(f"[info] znaleziono {len(glass_events)} glassbreak, "
          f"{len(hardneg_events)} gunshot/babycry (przed subsamplingiem)")

    if hard_negative_sample is not None and len(hardneg_events) > hard_negative_sample:
        random.Random(seed).shuffle(hardneg_events)
        hardneg_events = hardneg_events[:hard_negative_sample]
        print(f"[info] subsampling trudnych negatywów do {hard_negative_sample}")

    def _write_all(events, dest_dir: Path, tag: str) -> int:
        n_written = 0
        cache_path, cache_y, cache_sr = None, None, None
        for i, (audio_path, start, end) in enumerate(events):
            if audio_path != cache_path:
                cache_y, cache_sr = sf.read(audio_path, dtype="float32")
                cache_path = audio_path
            seg = _extract_segment(cache_y, cache_sr, start, end, pad)
            if len(seg) < int(0.05 * cache_sr):  # < 50ms — bezużyteczne
                continue
            stem = Path(audio_path).stem
            out_path = dest_dir / f"{tag}_{i:05d}_{stem}_{start:.2f}-{end:.2f}.wav"
            sf.write(out_path, seg, cache_sr)
            n_written += 1
        return n_written

    n_glass = _write_all(glass_events, glass_out, "voiceglass")
    n_hardneg = _write_all(hardneg_events, hardneg_out, "voicehardneg")

    print(f"[ok] zapisano {n_glass} klipów glass -> {glass_out}")
    print(f"[ok] zapisano {n_hardneg} klipów hard-negative -> {hardneg_out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotation", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", default="./voice_extracted")
    ap.add_argument("--pad", type=float, default=0.3, help="padding w sekundach z każdej strony")
    ap.add_argument("--hard-negative-sample", type=int, default=None,
                    help="ogranicz liczbę wyciętych gunshot/babycry (domyślnie: wszystkie)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    extract(args.annotation, args.audio, args.out, pad=args.pad,
            hard_negative_sample=args.hard_negative_sample, seed=args.seed)


if __name__ == "__main__":
    main()
