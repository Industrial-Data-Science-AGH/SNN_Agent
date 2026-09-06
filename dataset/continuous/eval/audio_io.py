"""
audio_io.py — wczytywanie i normalizacja audio do standardu zbioru.

Standard przyjęty jako DOMYŚLNY (do potwierdzenia przez Patryka — patrz
kryterium akceptacji "Format zaakceptowany przez Patryka"):

    sample rate : 44100 Hz
    kanały      : 1 (mono)
    format      : PCM_16

Uzasadnienie: to parametry, na jakich osadzony jest cały manifest v2.0.0
(dataset/versions/v2.0.0/stats.md, sekcja "Parametry audio": {44100: 10853},
{1: 10853}, {'PCM_16': 10853}). ESC-50 bywa 44100/mono, ale nie jest to tu
zakładane bezkrytycznie — każdy plik wejściowy jest jawnie resamplowany
i przetwarzany do mono niezależnie od natywnych parametrów.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 44100
TARGET_CHANNELS = 1
TARGET_SUBTYPE = "PCM_16"


@dataclass(frozen=True)
class AudioStandard:
    sample_rate: int = TARGET_SR
    channels: int = TARGET_CHANNELS
    subtype: str = TARGET_SUBTYPE


def load_audio_mono(path: str, standard: AudioStandard = AudioStandard()) -> np.ndarray:
    """Wczytuje plik audio i sprowadza go do float32 mono @ standard.sample_rate.

    Zwraca próbki w zakresie [-1, 1]. Rzuca FileNotFoundError jeśli pliku nie ma
    (jawnie, żeby nie powtórzyć błędu 'build-manifest pomija brakujące pliki').
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"brak pliku audio: {path}")

    data, sr = sf.read(path, dtype="float32", always_2d=True)  # [T, C]
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    if sr != standard.sample_rate:
        # resample_poly z dokładnym stosunkiem próbkowań (gcd-redukcja wewnątrz)
        from math import gcd
        g = gcd(sr, standard.sample_rate)
        up, down = standard.sample_rate // g, sr // g
        data = resample_poly(data, up, down).astype(np.float32)

    return data


def write_audio(path: str, samples: np.ndarray, standard: AudioStandard = AudioStandard()) -> None:
    """Zapisuje mono float32 [-1, 1] jako PCM_16 WAV @ standard.sample_rate."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    sf.write(path, clipped, standard.sample_rate, subtype=standard.subtype)


def peak_normalize(samples: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    """Skaluje sygnał tak, by szczyt amplitudy wynosił target_peak. Cisza -> bez zmian."""
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak < 1e-9:
        return samples
    return samples * (target_peak / peak)
