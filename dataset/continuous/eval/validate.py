"""
validate.py — programowa weryfikacja kryteriów akceptacji na gotowej parze
(audio, manifest). Używane przez testy automatyczne oraz do ręcznej kontroli
jakości wygenerowanych datasetów.
"""
from __future__ import annotations

import os

import soundfile as sf

from .manifest import load_manifest


class ValidationError(AssertionError):
    pass


def validate_pair(audio_path: str, manifest_path: str) -> dict:
    """Sprawdza:
      - dokładnie 5 zdarzeń w manifeście,
      - brak nakładania się zdarzeń,
      - wszystkie zdarzenia mieszczą się w [0, duration_s],
      - deklarowana długość audio zgadza się z rzeczywistym plikiem WAV,
      - sha256 w manifeście zgadza się z plikiem na dysku.

    Zwraca manifest (dict) przy sukcesie, rzuca ValidationError przy błędzie.
    """
    manifest = load_manifest(manifest_path)

    events = manifest["events"]
    if len(events) != 5:
        raise ValidationError(f"oczekiwano 5 zdarzeń, jest {len(events)}")

    events_sorted = sorted(events, key=lambda e: e["start_s"])
    for a, b in zip(events_sorted, events_sorted[1:]):
        if a["end_s"] > b["start_s"]:
            raise ValidationError(
                f"nakładające się zdarzenia: idx {a['index']} "
                f"[{a['start_s']}, {a['end_s']}] i idx {b['index']} "
                f"[{b['start_s']}, {b['end_s']}]"
            )

    declared_duration = manifest["audio"]["duration_s"]
    for e in events:
        if e["start_s"] < 0 or e["end_s"] > declared_duration:
            raise ValidationError(
                f"zdarzenie idx {e['index']} poza zakresem strumienia "
                f"[0, {declared_duration}]: [{e['start_s']}, {e['end_s']}]"
            )

    if not os.path.exists(audio_path):
        raise ValidationError(f"plik audio nie istnieje: {audio_path}")

    info = sf.info(audio_path)
    actual_duration = info.frames / info.samplerate
    if abs(actual_duration - declared_duration) > 0.05:
        raise ValidationError(
            f"długość audio ({actual_duration:.3f}s) nie zgadza się z "
            f"manifestem ({declared_duration:.3f}s)"
        )
    if info.samplerate != manifest["audio"]["sample_rate"]:
        raise ValidationError(
            f"sample rate audio ({info.samplerate}) != manifest "
            f"({manifest['audio']['sample_rate']})"
        )
    if info.channels != manifest["audio"]["channels"]:
        raise ValidationError(
            f"liczba kanałów audio ({info.channels}) != manifest "
            f"({manifest['audio']['channels']})"
        )

    from .stream_builder import sha256_of_file
    actual_sha = sha256_of_file(audio_path)
    if actual_sha != manifest["audio"]["sha256"]:
        raise ValidationError(
            f"sha256 audio nie zgadza się z manifestem "
            f"({actual_sha} != {manifest['audio']['sha256']})"
        )

    return manifest
