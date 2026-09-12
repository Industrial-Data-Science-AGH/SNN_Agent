"""
manifest.py — kontrakt manifestu ciągłego datasetu ewaluacyjnego.

UWAGA: format NIE był jeszcze zaakceptowany przez Marcela (master pipeline)
ani Patryka (standard datasetu). Projekt roboczy — zmiana formatu = tylko
ten plik, generator bez zmian.

Schemat (manifest_schema_version="1.1.0"):

{
  "manifest_schema_version": "1.1.0",
  "generator_version": "1.0.0",
  "generated_utc": "2026-09-01T12:00:00+00:00",
  "seed": 42,
  "git_commit": "abc1234" | null,
  "audio": {
    "path": "continuous_eval_seed42.wav",
    "sha256": "...",
    "sample_rate": 44100,
    "channels": 1,
    "subtype": "PCM_16",
    "duration_s": 600.0
  },
  "config": {
    "glassbreak_mode": "clean" | "background",
    "min_gap_s": 2.0,
    "edge_margin_s": 1.0,
    "warmup_s": 30.0,
    "warmup_excluded_from_fa": true,
    "event_gain_db_range": [-3.0, 3.0],
    "background_gain_db": -2.87,
    "background_dirs": ["data/ESC-50-master/audio"],
    "glass_audio_root": "dataset/clean/clean/audio",
    "glass_allowed_stems_file": "dataset/clean/clean/target/synthetic_target_test.txt" | null,
    "overlap_check": {
      "train_stems_files": ["dataset/clean/clean/source/synthetic_source_training.txt"],
      "result": {"synthetic_source_training.txt": []}
    }
  },
  "events": [
    {
      "index": 0,
      "start_s": 47.23,       -- zawsze >= warmup_s
      "end_s": 48.09,
      "duration_s": 0.86,
      "source_stem": "synthetic_014",
      "source_start_s": 4.00,
      "source_end_s": 5.36,
      "is_contaminated": false,
      "overlapping_labels": [],
      "gain_db": 1.2
    },
    ... dokładnie 5 wpisów, posortowane rosnąco po start_s ...
  ],
  "background_segments": [
    {
      "path": "data/ESC-50-master/audio/1-100032-A-0.wav",
      "source": "ESC-50",
      "kind": "animal",           -- z meta/esc50.csv, jedna z: animal/stationary/speech/loud_event
      "stream_start_s": 0.0,
      "stream_end_s": 5.02
    },
    ...
  ]
}

Jak Marcel liczy metryki:
  - event recall: dla i-tego wpisu w "events", czy detektor podniósł alarm
    w [start_s, end_s + tolerancja]
  - latency: czas między start_s a momentem pierwszego alarmu w tym oknie
  - false alarms/h: alarmy poza wszystkimi oknami [start_s, end_s],
    liczone NA ODCINKU [warmup_s, duration_s] (warmup wyłączony),
    podzielone przez (duration_s - warmup_s) / 3600
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Sequence

from .stream_builder import GENERATOR_VERSION, GeneratedStream, git_commit_short, sha256_of_file

MANIFEST_SCHEMA_VERSION = "1.1.0"


def build_manifest_dict(
    *,
    stream: GeneratedStream,
    audio_path: str,
    seed: int,
    glassbreak_mode: str,
    min_gap_s: float,
    warmup_s: float,
    end_margin_s: float,
    event_gain_db_range: tuple[float, float],
    background_dirs: Sequence[str],
    glass_audio_root: str,
    glass_allowed_stems_files: str | None,
    overlap_check: dict | None = None,
) -> dict:
    if len(stream.events) != 5:
        raise AssertionError(
            f"manifest wymaga dokładnie 5 zdarzeń, otrzymano {len(stream.events)}"
        )

    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "git_commit": git_commit_short(),
        "audio": {
            "path": os.path.basename(audio_path),
            "sha256": sha256_of_file(audio_path),
            "sample_rate": stream.sample_rate,
            "channels": 1,
            "subtype": "PCM_16",
            "duration_s": round(stream.audio.size / stream.sample_rate, 6),
        },
        "config": {
            "glassbreak_mode": glassbreak_mode,
            "min_gap_s": min_gap_s,
            "end_margin_s": end_margin_s,
            "warmup_s": warmup_s,
            "warmup_excluded_from_fa": True,
            "event_gain_db_range": list(event_gain_db_range),
            "background_gain_db": round(stream.background_gain_db, 3),
            "background_dirs": [os.path.relpath(d) for d in background_dirs],
            "glass_audio_root": os.path.relpath(glass_audio_root),
            "glass_allowed_stems_file": (
                os.path.relpath(glass_allowed_stems_files)
                if glass_allowed_stems_files else None
            ),
            "overlap_check": overlap_check or {},
        },
        "events": [
            {"index": i, **e.to_manifest_dict()}
            for i, e in enumerate(stream.events)
        ],
        "background_segments": stream.background_segments,
    }


def write_manifest(manifest: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)


def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)