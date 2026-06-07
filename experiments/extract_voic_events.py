"""
extract_voic_events.py - Split VOICe synthetic mixes into per-event WAV segments.

Reads synthetic_NNN.wav from positive/, slices by dataset/clean/annotation/*.txt,
writes glassbreak -> positive/, babycry/gunshot -> negative/, moves originals
to dataset/clean/audio/.

Usage:
    python experiments/extract_voic_events.py
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple


LABEL_SUFFIX = {
    "glassbreak": "gb",
    "babycry": "bc",
    "gunshot": "gs",
}

MIN_SEGMENT_SEC = 0.01

NEGATIVE_LABELS = frozenset({"babycry", "gunshot"})


@dataclass
class Event:
    onset: float
    offset: float
    label: str


def parse_annotation(path: str) -> List[Event]:
    events: List[Event] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                continue
            onset, offset, label = parts
            events.append(Event(float(onset), float(offset), label.strip()))
    return events


def intervals_overlap(a_on: float, a_off: float, b_on: float, b_off: float) -> bool:
    return a_on < b_off and b_on < a_off


def subtract_intervals(
    segment: Tuple[float, float],
    blocks: List[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    parts = [segment]
    for b_on, b_off in blocks:
        next_parts: List[Tuple[float, float]] = []
        for s_on, s_off in parts:
            if not intervals_overlap(s_on, s_off, b_on, b_off):
                next_parts.append((s_on, s_off))
                continue
            if s_on < b_on:
                next_parts.append((s_on, b_on))
            if b_off < s_off:
                next_parts.append((b_off, s_off))
        parts = next_parts
    return [(s_on, s_off) for s_on, s_off in parts if (s_off - s_on) >= MIN_SEGMENT_SEC]


def glassbreak_intervals(events: List[Event]) -> List[Tuple[float, float]]:
    return [(e.onset, e.offset) for e in events if e.label == "glassbreak"]


def output_dirs(args: argparse.Namespace) -> Dict[str, str]:
    return {
        "glassbreak": args.positive_dir,
        "babycry": args.negative_dir,
        "gunshot": args.negative_dir,
    }


def extract_segment_ffmpeg(
    wav_path: str,
    onset: float,
    offset: float,
    out_path: str,
) -> bool:
    duration = offset - onset
    if duration < MIN_SEGMENT_SEC:
        return False

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{onset}",
        "-i",
        wav_path,
        "-t",
        f"{duration}",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return True


def write_segment(
    wav_path: str,
    onset: float,
    offset: float,
    out_path: str,
    force: bool,
) -> bool:
    if os.path.exists(out_path) and not force:
        return True
    return extract_segment_ffmpeg(wav_path, onset, offset, out_path)


def process_file(
    wav_path: str,
    ann_path: str,
    dirs: Dict[str, str],
    force: bool,
    labels: frozenset[str],
) -> Tuple[Dict[str, int], int]:
    events = parse_annotation(ann_path)
    if not events:
        print(f"WARN: no events in {ann_path}", file=sys.stderr)
        return {}, 0

    stem = os.path.splitext(os.path.basename(wav_path))[0]
    counters: Dict[str, int] = {label: 0 for label in LABEL_SUFFIX}
    skipped = 0
    gb_blocks = glassbreak_intervals(events)

    for event in events:
        label = event.label
        if label not in LABEL_SUFFIX or label not in labels:
            continue

        if label in NEGATIVE_LABELS:
            segments = subtract_intervals((event.onset, event.offset), gb_blocks)
            if not segments:
                skipped += 1
                continue
        else:
            segments = [(event.onset, event.offset)]

        for onset, offset in segments:
            counters[label] += 1
            idx = counters[label]
            suffix = LABEL_SUFFIX[label]
            out_dir = dirs[label]
            out_name = f"{stem}_{suffix}_{idx:03d}.wav"
            out_path = os.path.join(out_dir, out_name)

            try:
                ok = write_segment(wav_path, onset, offset, out_path, force)
            except subprocess.CalledProcessError:
                skipped += 1
                counters[label] -= 1
                if os.path.exists(out_path):
                    os.remove(out_path)
                continue

            if not ok:
                skipped += 1
                counters[label] -= 1
                if os.path.exists(out_path):
                    os.remove(out_path)

    return counters, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract VOICe event segments from synthetic mixes.")
    parser.add_argument("--source-dir", default="positive")
    parser.add_argument("--annotation-dir", default="dataset/clean/annotation")
    parser.add_argument("--positive-dir", default="positive")
    parser.add_argument("--negative-dir", default="negative")
    parser.add_argument("--clean-audio-dir", default="dataset/clean/audio")
    parser.add_argument("--move-originals", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--labels",
        default="glassbreak,babycry,gunshot",
        help="Comma-separated labels to extract (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing segment files")
    parser.add_argument(
        "--purge-negative-synthetic",
        action="store_true",
        help="Delete existing synthetic_*_{bc,gs}_*.wav in negative-dir before extraction",
    )
    args = parser.parse_args()
    labels = frozenset(x.strip() for x in args.labels.split(",") if x.strip())
    unknown = labels - set(LABEL_SUFFIX)
    if unknown:
        print(f"Unknown labels: {', '.join(sorted(unknown))}", file=sys.stderr)
        sys.exit(1)

    for d in (args.positive_dir, args.negative_dir, args.clean_audio_dir):
        os.makedirs(d, exist_ok=True)

    if args.purge_negative_synthetic:
        removed = 0
        for pattern in ("synthetic_*_bc_*.wav", "synthetic_*_gs_*.wav"):
            for path in glob.glob(os.path.join(args.negative_dir, pattern)):
                os.remove(path)
                removed += 1
        print(f"Purged {removed} synthetic negative segments from {args.negative_dir}/")

    dirs = output_dirs(args)
    wav_files = sorted(glob.glob(os.path.join(args.source_dir, "synthetic_*.wav")))
    if not wav_files:
        print(f"No synthetic_*.wav files found in {args.source_dir}", file=sys.stderr)
        sys.exit(1)

    totals: Dict[str, int] = {label: 0 for label in LABEL_SUFFIX}
    total_skipped = 0
    moved = 0
    missing_ann = 0

    for i, wav_path in enumerate(wav_files, start=1):
        stem = os.path.splitext(os.path.basename(wav_path))[0]
        ann_path = os.path.join(args.annotation_dir, f"{stem}.txt")
        if not os.path.isfile(ann_path):
            print(f"WARN: missing annotation for {wav_path}", file=sys.stderr)
            missing_ann += 1
            continue

        counters, skipped = process_file(wav_path, ann_path, dirs, args.force, labels)
        for label, count in counters.items():
            totals[label] += count
        total_skipped += skipped

        if args.move_originals and counters:
            dest = os.path.join(args.clean_audio_dir, os.path.basename(wav_path))
            if os.path.exists(dest) and not args.force:
                print(f"WARN: already exists, not moving {wav_path} -> {dest}", file=sys.stderr)
            else:
                shutil.move(wav_path, dest)
                moved += 1

        print(f"[{i}/{len(wav_files)}] {stem}: gb={counters.get('glassbreak', 0)} bc={counters.get('babycry', 0)} gs={counters.get('gunshot', 0)}")

    print("Extraction complete.")
    print(f"  Source files processed: {len(wav_files) - missing_ann}")
    print(f"  Missing annotations:    {missing_ann}")
    print(f"  glassbreak segments:    {totals['glassbreak']} -> {args.positive_dir}/")
    print(f"  babycry segments:       {totals['babycry']} -> {args.negative_dir}/")
    print(f"  gunshot segments:       {totals['gunshot']} -> {args.negative_dir}/")
    print(f"  Skipped events:         {total_skipped}")
    if args.move_originals:
        print(f"  Originals moved:        {moved} -> {args.clean_audio_dir}/")


if __name__ == "__main__":
    main()
