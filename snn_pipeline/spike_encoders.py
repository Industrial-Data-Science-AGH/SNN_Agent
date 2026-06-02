# -*- coding: utf-8 -*-
"""
Enkodery spike'ów — konwersja sygnału audio na ciągi impulsów (spike trains).

FIX v2: Naprawiony TTFS encoder — stara wersja generowała 1 spike per RMS frame,
co dawało saturation (N2 clip=[1.0, 1.0]) bo każdy frame produkował spike.
Nowa wersja generuje burst proportionalny do peak energy, z poprawnym
rozkładem spike'ów w oknie czasowym.

Implementuje dwa schematy enkodowania:
1. Rate Coding — amplituda → częstotliwość spike'ów [5-200Hz]
2. TTFS (Time-To-First-Spike) — peak energy → pozycja burst-u w oknie

Oba warianty produkują tensory binarne (0/1) o wymiarach [n_channels, n_timesteps].
"""

import numpy as np
import torch
from typing import Tuple

from snn_pipeline.config import AUDIO_CONFIG, RANDOM_SEED


class RateCodingEncoder:
    """Enkoder Rate Coding: amplituda energii → częstotliwość spike'ów.

    Wyższa energia = wyższa częstotliwość spike'ów w oknie czasowym.
    Zakres: [rate_min_hz, rate_max_hz] Hz.

    FIX v2: rng tworzone per-call z losowym seedem (nie globalnym)
    żeby kolejne wywołania dawały różne spike trains.

    Attributes:
        rate_min: Minimalna częstotliwość spike'ów (Hz).
        rate_max: Maksymalna częstotliwość spike'ów (Hz).
        dt: Krok czasowy symulacji (sekundy).

    Przykład:
        >>> encoder = RateCodingEncoder()
        >>> energy = np.array([0.0, 0.5, 1.0, 0.3])
        >>> spikes = encoder.encode(energy, n_timesteps=100)
        >>> print(spikes.shape)  # (4, 100)
    """

    def __init__(
        self,
        rate_min: float = AUDIO_CONFIG.rate_min_hz,
        rate_max: float = AUDIO_CONFIG.rate_max_hz,
        dt: float = AUDIO_CONFIG.spike_duration_s,
    ) -> None:
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.dt = dt
        self._call_count = 0

    def encode(
        self,
        energy_normalized: np.ndarray,
        n_timesteps: int = 100,
        jitter_pct: float = 0.0,
    ) -> torch.Tensor:
        """Enkoduje znormalizowaną energię [0,1] na spike train metodą Rate Coding.

        Dla każdego kanału generuje stochastyczny spike train gdzie
        prawdopodobieństwo spike'a w każdym kroku = rate × dt.

        Args:
            energy_normalized: Tablica energii [0,1], kształt (n_channels,).
            n_timesteps: Liczba kroków czasowych.
            jitter_pct: Opcjonalny jitter timing (%).

        Returns:
            Tensor binarny (0/1) kształtu (n_channels, n_timesteps).
        """
        energy_clipped = np.clip(energy_normalized, 0.0, 1.0)

        # Częstotliwość = rate_min + energy × (rate_max - rate_min)
        frequencies = self.rate_min + energy_clipped * (self.rate_max - self.rate_min)

        # Prawdopodobieństwo spike'a na krok = frequency × dt
        spike_probs = frequencies * self.dt
        spike_probs = np.clip(spike_probs, 0.0, 1.0)

        # FIX: różny seed per wywołanie (stary użyty stały seed = identyczne wyniki)
        self._call_count += 1
        rng = np.random.default_rng(RANDOM_SEED + self._call_count)

        # Spike train (Poisson)
        random_vals = rng.random((len(energy_clipped), n_timesteps))
        spikes = (random_vals < spike_probs[:, np.newaxis]).astype(np.float32)

        # Opcjonalny jitter
        if jitter_pct > 0:
            jitter = rng.normal(0, jitter_pct / 100.0, size=spikes.shape)
            shifted = spikes + jitter
            spikes = (shifted > 0.5).astype(np.float32)

        return torch.tensor(spikes, dtype=torch.float32)

    def encode_single(
        self, energy: float, n_timesteps: int = 100
    ) -> torch.Tensor:
        """Enkoduje pojedynczą wartość energii na spike train.

        Args:
            energy: Znormalizowana energia [0,1].
            n_timesteps: Liczba kroków czasowych.

        Returns:
            Tensor binarny kształtu (n_timesteps,).
        """
        result = self.encode(np.array([energy]), n_timesteps)
        return result.squeeze(0)


class TTFSEncoder:
    """Enkoder Time-To-First-Spike (Fixed v2).

    FIX v2 — KOMPLETNIE PRZEPISANY:
    Stara wersja generowała 1 spike per RMS frame → N2 zawsze saturowane bo
    KAŻDY frame produkował spike (spike_rate = n_aktywnych_frames / n_timesteps ≈ 1.0).

    Nowa wersja:
    1. Bierze ENVELOPE energii (nie per-frame)
    2. Generuje BURST proporcjonalny do peak energy
    3. Pozycja burst-u w oknie = TTFS delay
    4. Liczba spike'ów w burst = proporcjonalna do amplitudy

    Glass break: wysoka energia → krótki burst na początku okna (delay ≈ 0)
    Szum HVAC: niska energia → pojedynczy spike na końcu okna lub brak

    Attributes:
        window_ms: Szerokość okna TTFS (milisekundy).
        dt: Krok czasowy (sekundy).
        burst_max_spikes: Max spike'ów w burst (dla energy=1.0).

    Przykład:
        >>> encoder = TTFSEncoder()
        >>> energy = np.array([1.0, 0.5, 0.0])
        >>> spikes = encoder.encode(energy, n_timesteps=100)
        >>> # energy=1.0 → burst 5 spike'ów na t=0..4
        >>> # energy=0.5 → 2-3 spike'i na t=10..12
        >>> # energy=0.0 → 0 spike'ów
    """

    def __init__(
        self,
        window_ms: float = AUDIO_CONFIG.ttfs_window_ms,
        dt: float = AUDIO_CONFIG.spike_duration_s,
        burst_max_spikes: int = 5,
    ) -> None:
        self.window_ms = window_ms
        self.dt = dt
        self.n_steps_in_window = int(window_ms / (dt * 1000))
        self.burst_max_spikes = burst_max_spikes
        self._call_count = 0

    def encode(
        self,
        energy_normalized: np.ndarray,
        n_timesteps: int = 100,
        jitter_pct: float = 0.0,
    ) -> torch.Tensor:
        """Enkoduje energy envelope na spike train z burst pattern.

        FIX v2: Zamiast 1-spike-per-frame, generuje burst proporcjonalny
        do peak energy. Niska energia (<0.1) → 0 spike'ów (poniżej progu szumu).

        Args:
            energy_normalized: Tablica energii [0,1], kształt (n_channels,).
            n_timesteps: Liczba kroków czasowych.
            jitter_pct: Opcjonalny jitter timing (%).

        Returns:
            Tensor binarny (0/1) kształtu (n_channels, n_timesteps).
        """
        energy_clipped = np.clip(energy_normalized, 0.0, 1.0)

        self._call_count += 1
        rng = np.random.default_rng(RANDOM_SEED + self._call_count + 10000)

        spikes = np.zeros((len(energy_clipped), n_timesteps), dtype=np.float32)

        for i, e in enumerate(energy_clipped):
            # Próg szumu: energia < 0.05 → brak spike'a (filtruje tło)
            if e < 0.05:
                continue

            # TTFS delay: im wyższa energia, tym wcześniejszy burst
            delay = int(self.n_steps_in_window * (1.0 - e))
            delay = max(0, min(delay, n_timesteps - 1))

            # Liczba spike'ów w burst proporcjonalna do energii
            n_burst = max(1, int(e * self.burst_max_spikes))

            # Wstaw burst (kolejne spike'i od pozycji delay)
            for s in range(n_burst):
                spike_pos = delay + s
                if spike_pos < n_timesteps:
                    spikes[i, spike_pos] = 1.0

        # Opcjonalny jitter
        if jitter_pct > 0:
            jitter_mask = rng.normal(0, jitter_pct / 100.0, size=spikes.shape)
            jittered = spikes + jitter_mask
            spikes = (jittered > 0.5).astype(np.float32)

        return torch.tensor(spikes, dtype=torch.float32)

    def encode_single(
        self, energy: float, n_timesteps: int = 100
    ) -> torch.Tensor:
        """Enkoduje pojedynczą wartość energii na spike train TTFS.

        Args:
            energy: Znormalizowana energia [0,1].
            n_timesteps: Liczba kroków czasowych.

        Returns:
            Tensor binarny kształtu (n_timesteps,).
        """
        result = self.encode(np.array([energy]), n_timesteps)
        return result.squeeze(0)


def encode_audio_to_spikes(
    energy_envelope: np.ndarray,
    encoding: str = "ttfs",
    n_timesteps: int = 100,
    jitter_pct: float = 0.0,
) -> torch.Tensor:
    """Convenience function — enkoduje envelope energii audio na spike train.

    FIX v2: Zamiast per-frame encoding (który saturował N2), bierze
    statystyki z envelope (peak, mean, std) i produkuje 3-kanałowy tensor:
    - Kanał 0: peak energy → TTFS/Rate (detekcja impulsów)
    - Kanał 1: mean energy → TTFS/Rate (baseline poziomu)
    - Kanał 2: std energy → TTFS/Rate (zmienność = burst detection)

    Args:
        energy_envelope: Tablica RMS energy per frame, kształt (n_frames,).
        encoding: "ttfs" lub "rate".
        n_timesteps: Liczba kroków czasowych.
        jitter_pct: Jitter (%).

    Returns:
        Tensor spike'ów kształtu (3, n_timesteps).

    Przykład:
        >>> envelope = np.array([0.1, 0.3, 0.8, 0.9, 0.7, 0.2, 0.1])
        >>> spikes = encode_audio_to_spikes(envelope, encoding="ttfs")
        >>> print(spikes.shape)  # (3, 100)
    """
    # Statystyki z envelope (zamiast per-frame encoding)
    if len(energy_envelope) == 0:
        return torch.zeros(3, n_timesteps)

    peak = float(np.max(energy_envelope))
    mean = float(np.mean(energy_envelope))
    std = float(np.std(energy_envelope))

    # Normalizacja std (maksymalnie = 0.5 bo to max std dla [0,1])
    std_norm = min(std / 0.5, 1.0)

    features = np.array([peak, mean, std_norm], dtype=np.float32)

    if encoding == "ttfs":
        encoder = TTFSEncoder()
    else:
        encoder = RateCodingEncoder()

    return encoder.encode(features, n_timesteps=n_timesteps, jitter_pct=jitter_pct)


def compare_encoders(
    energy_values: np.ndarray,
    n_timesteps: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Porównuje oba enkodery na tych samych danych energii.

    Args:
        energy_values: Tablica energii [0,1].
        n_timesteps: Liczba kroków czasowych.

    Returns:
        Krotka (rate_spikes, ttfs_spikes).

    Przykład:
        >>> energy = np.array([0.2, 0.5, 0.8, 1.0])
        >>> rate, ttfs = compare_encoders(energy)
    """
    rc_encoder = RateCodingEncoder()
    ttfs_encoder = TTFSEncoder()

    rate_spikes = rc_encoder.encode(energy_values, n_timesteps)
    ttfs_spikes = ttfs_encoder.encode(energy_values, n_timesteps)

    return rate_spikes, ttfs_spikes
