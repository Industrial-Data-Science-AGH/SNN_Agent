"""
manifest.py — proponowany kontrakt manifestu ciągłego datasetu ewaluacyjnego.

UWAGA (patrz README, sekcja "Kontrakt manifestu — do akceptacji"): ten format
NIE był jeszcze zaakceptowany przez Marcela (konsument, master pipeline) ani
Patryka (standard datasetu). To jest projekt roboczy, zaprojektowany tak, by:

  - dało się z niego policzyć event recall, false alarms/h i latency detekcji
    bez ponownego parsowania audio,
  - był czytelny bez zależności od torcha/SNN-specyficznych bibliotek
    (zwykły json.load wystarcza),
  - niósł provenance (wersja generatora, seed, sha256 audio) wystarczające do
    odtworzenia przebiegu i do stwierdzenia "to jest DOKŁADNIE ten sam plik".

Schemat (manifest_schema_version="1.0.0"):

{
  "manifest_schema_version": "1.0.0",
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
    "event_gain_db_range": [-3.0, 3.0],
    "background_gain_db": -2.87,
    "background_dirs": ["data/ESC-50-master/audio"],
    "glass_audio_root": "dataset/clean/clean/audio",
    "glass_allowed_stems_file": "dataset/clean/clean/target/synthetic_target_test.txt" | null
  },
  "events": [
    {
      "index": 0,
      "start_s": 12.34,
      "end_s": 13.10,
      "duration_s": 0.76,
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
    {"path": "...", "source": "ESC-50-master", "stream_start_s": 0.0, "stream_end_s": 5.02},
    ...
  ]
}

Konsument (master pipeline Marcela) liczy metryki tak:
  - event recall: dla i-tego wpisu w "events", czy detektor podniósł alarm
    w [start_s, end_s + tolerancja]
  - latency: czas między start_s a momentem pierwszego alarmu w tym oknie
  - false alarms/h: każdy alarm poza wszystkimi oknami [start_s, end_s] (+
    ewentualny margines) na godzinę długości audio.duration_s
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Sequence

from .stream_builder import GENERATOR_VERSION, GeneratedStream, git_commit_short, sha256_of_file

MANIFEST_SCHEMA_VERSION = "1.0.0"


def build_manifest_dict(
    *,
    stream: GeneratedStream,
    audio_path: str,
    seed: int,
    glassbreak_mode: str,
    min_gap_s: float,
    edge_margin_s: float,
    event_gain_db_range: tuple[float, float],
    background_dirs: Sequence[str],
    glass_audio_root: str,
    glass_allowed_stems_file: str | None,
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
            "edge_margin_s": edge_margin_s,
            "event_gain_db_range": list(event_gain_db_range),
            "background_gain_db": round(stream.background_gain_db, 3),
            "background_dirs": [os.path.relpath(d) for d in background_dirs],
            "glass_audio_root": os.path.relpath(glass_audio_root),
            "glass_allowed_stems_file": (
                os.path.relpath(glass_allowed_stems_file) if glass_allowed_stems_file else None
            ),
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
