# -*- coding: utf-8 -*-
"""
Pipeline danych — pobieranie ESC-50, preprocessing audio, augmentacja, budowanie datasetu.

Odpowiada za kompletny łańcuch: .wav → filtr pasmowy → ekstrakcja cech → spike encoding → tensor.
Augmentacja rozbudowuje 40 próbek glass_break do 400+ przez time stretch, pitch shift,
additive noise, volume scaling i syntetyczny Room Impulse Response.
"""

import os
import zipfile
import urllib.request
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from scipy.signal import butter, sosfilt, fftconvolve
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from snn_pipeline.config import (
    AUDIO_CONFIG,
    ESC50_GLASS_BREAKING_CLASS,
    ESC50_URL,
    PATH_CONFIG,
    RANDOM_SEED,
    TRAIN_CONFIG,
    DEVICE,
)
from snn_pipeline.spike_encoders import RateCodingEncoder, TTFSEncoder


# =============================================================================
# POBIERANIE DATASETU ESC-50
# =============================================================================

def download_esc50(data_dir: Optional[Path] = None) -> Path:
    """Pobiera dataset ESC-50 z GitHub jeśli jeszcze nie istnieje.

    Dataset waży ~600MB. Pobieranie wymaga internetu.
    Po pobraniu rozpakowuje ZIP i zwraca ścieżkę do katalogu z plikami .wav.

    Args:
        data_dir: Katalog docelowy. Domyślnie: config.PATH_CONFIG.data_dir.

    Returns:
        Ścieżka do katalogu ESC-50 z plikami audio.

    Przykład:
        >>> esc50_path = download_esc50()
        >>> print(f"ESC-50 w: {esc50_path}")
    """
    if data_dir is None:
        data_dir = PATH_CONFIG.data_dir

    esc50_dir = data_dir / "ESC-50-master"
    audio_dir = esc50_dir / "audio"

    if audio_dir.exists() and len(list(audio_dir.glob("*.wav"))) > 0:
        print(f"[INFO] ESC-50 już istnieje w {esc50_dir} — pomijam pobieranie.")
        return esc50_dir

    zip_path = data_dir / "ESC-50-master.zip"
    if not zip_path.exists():
        print(f"[INFO] Pobieram ESC-50 z {ESC50_URL}...")
        print("[INFO] To może potrwać ~2 minuty (600MB)...")
        urllib.request.urlretrieve(ESC50_URL, str(zip_path))
        print("[INFO] Pobieranie zakończone.")

    print("[INFO] Rozpakowuję archiwum ZIP...")
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(data_dir))
    print(f"[INFO] ESC-50 rozpakowany do {esc50_dir}")

    # Usuń ZIP żeby oszczędzić miejsce
    zip_path.unlink(missing_ok=True)

    return esc50_dir


# =============================================================================
# LOADING I PREPROCESSING AUDIO
# =============================================================================

def load_audio(
    path: str,
    sr: int = AUDIO_CONFIG.sample_rate,
    duration: Optional[float] = None,
) -> Tuple[np.ndarray, int]:
    """Ładuje plik .wav i resampluje do docelowej częstotliwości.

    Args:
        path: Ścieżka do pliku .wav.
        sr: Docelowa częstotliwość próbkowania (Hz).
        duration: Opcjonalny czas trwania do załadowania (sekundy).

    Returns:
        Krotka (audio_array, sample_rate).

    Przykład:
        >>> audio, sr = load_audio("glass_break_001.wav")
        >>> print(f"Załadowano {len(audio)} próbek @ {sr}Hz")
    """
    audio, sr_out = librosa.load(path, sr=sr, duration=duration, mono=True)
    return audio, sr_out


def bandpass_filter(
    audio: np.ndarray,
    sr: int = AUDIO_CONFIG.sample_rate,
    low_freq: float = AUDIO_CONFIG.bandpass_low,
    high_freq: float = AUDIO_CONFIG.bandpass_high,
    order: int = 4,
) -> np.ndarray:
    """Stosuje filtr pasmowy Butterwortha na sygnale audio.

    Przepuszcza częstotliwości w zakresie glass break (500Hz–8000Hz),
    odcinając szumy niskotonowe (HVAC) i ultrawysokoczęstotliwościowe.

    Args:
        audio: Sygnał mono.
        sr: Częstotliwość próbkowania (Hz).
        low_freq: Dolna częstotliwość graniczna (Hz).
        high_freq: Górna częstotliwość graniczna (Hz).
        order: Rząd filtru Butterwortha.

    Returns:
        Odfiltrowany sygnał audio.

    Przykład:
        >>> filtered = bandpass_filter(audio, sr=22050, low_freq=500, high_freq=8000)
    """
    nyquist = sr / 2.0
    low = low_freq / nyquist
    high = high_freq / nyquist
    # Clamp do bezpiecznego zakresu
    low = max(low, 0.001)
    high = min(high, 0.999)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def extract_rms_energy(
    audio: np.ndarray,
    sr: int = AUDIO_CONFIG.sample_rate,
    window_ms: float = AUDIO_CONFIG.rms_window_ms,
    hop_ms: float = AUDIO_CONFIG.rms_hop_ms,
) -> np.ndarray:
    """Oblicza RMS energy w przesuwnych oknach czasowych.

    Args:
        audio: Sygnał audio (mono).
        sr: Częstotliwość próbkowania (Hz).
        window_ms: Szerokość okna (milisekundy).
        hop_ms: Krok pomiędzy oknami (milisekundy).

    Returns:
        Tablica RMS energy, kształt (n_frames,).

    Przykład:
        >>> energy = extract_rms_energy(audio, sr=22050, window_ms=10, hop_ms=5)
        >>> print(f"{len(energy)} okien energii")
    """
    frame_length = int(sr * window_ms / 1000.0)
    hop_length = int(sr * hop_ms / 1000.0)
    frame_length = max(frame_length, 1)
    hop_length = max(hop_length, 1)

    rms = librosa.feature.rms(
        y=audio, frame_length=frame_length, hop_length=hop_length
    )[0]
    return rms.astype(np.float32)


def extract_mfcc(
    audio: np.ndarray,
    sr: int = AUDIO_CONFIG.sample_rate,
    n_mfcc: int = AUDIO_CONFIG.n_mfcc,
) -> np.ndarray:
    """Oblicza MFCC (Mel-Frequency Cepstral Coefficients).

    Args:
        audio: Sygnał audio (mono).
        sr: Częstotliwość próbkowania (Hz).
        n_mfcc: Liczba współczynników MFCC.

    Returns:
        Tablica MFCC kształtu (n_mfcc, n_frames).

    Przykład:
        >>> mfcc = extract_mfcc(audio, sr=22050, n_mfcc=13)
        >>> print(f"MFCC shape: {mfcc.shape}")
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc.astype(np.float32)


def extract_features(
    audio: np.ndarray,
    sr: int = AUDIO_CONFIG.sample_rate,
    use_mfcc: bool = False,
) -> np.ndarray:
    """Główna funkcja ekstrakcji cech — RMS energy + opcjonalnie MFCC.

    Zwraca znormalizowaną energię [0,1] dla spike encoding.

    Args:
        audio: Sygnał audio (mono).
        sr: Częstotliwość próbkowania (Hz).
        use_mfcc: Czy dodać MFCC 13 współczynników (domyślnie False — prosty RMS).

    Returns:
        Tablica cech znormalizowana do [0,1], kształt (n_features, n_frames).

    Przykład:
        >>> features = extract_features(audio)
        >>> print(f"Feature shape: {features.shape}")
    """
    # Filtr pasmowy 500–8000 Hz (zakres glass break)
    filtered = bandpass_filter(audio, sr)

    # RMS energy
    rms = extract_rms_energy(filtered, sr)

    # Zmiana: absolutna normalizacja logarytmiczna zamiast per-sample!
    # Self-normalizacja sprawiała, że cisza i szum miały szczyt 1.0, przez co
    # TTFS nie mógł odróżnić tła od prawdziwych peaków.
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    
    # Mapowanie: tło (<-50dB) -> 0.0, szczyty (>-10dB) -> 1.0
    rms_norm = (rms_db + 50.0) / 40.0
    rms_norm = np.clip(rms_norm, 0.0, 1.0)

    if use_mfcc:
        mfcc = extract_mfcc(filtered, sr)
        # Normalizacja MFCC per-współczynnik
        mfcc_min = mfcc.min(axis=1, keepdims=True)
        mfcc_max = mfcc.max(axis=1, keepdims=True)
        mfcc_range = mfcc_max - mfcc_min
        mfcc_range[mfcc_range == 0] = 1.0
        mfcc_norm = (mfcc - mfcc_min) / mfcc_range
        # Łączymy RMS + MFCC: (1 + n_mfcc, n_frames)
        # Dopasuj długość
        min_len = min(rms_norm.shape[0], mfcc_norm.shape[1])
        features = np.vstack([rms_norm[np.newaxis, :min_len], mfcc_norm[:, :min_len]])
        return features.astype(np.float32)

    return rms_norm[np.newaxis, :].astype(np.float32)  # (1, n_frames)


# =============================================================================
# AUGMENTACJA DANYCH
# =============================================================================

def _generate_synthetic_rir(
    sr: int = AUDIO_CONFIG.sample_rate,
    rt60: float = 0.3,
    room_dim: Tuple[float, float, float] = (4.0, 3.0, 2.5),
) -> np.ndarray:
    """Generuje syntetyczny Room Impulse Response (RIR).

    Prosty model eksponentijalnego zanikania — symulacja małego pomieszczenia.

    Args:
        sr: Częstotliwość próbkowania.
        rt60: Czas pogłosu RT60 (sekundy).
        room_dim: Wymiary pomieszczenia (m) — tylko do obliczenia parametrów.

    Returns:
        Tablica z syntetycznym RIR.
    """
    rng = np.random.default_rng(RANDOM_SEED + 7)
    n_samples = int(sr * rt60)
    t = np.arange(n_samples) / sr
    # Eksponencjalny zanik
    decay = np.exp(-6.908 * t / rt60)  # -60dB w rt60
    # Losowe odbicia
    rir = rng.normal(0, 1, n_samples).astype(np.float32) * decay
    # Pierwszy sample = impuls bezpośredni
    rir[0] = 1.0
    # Normalizacja
    rir = rir / np.abs(rir).max()
    return rir


def augment_glass_break(
    audio: np.ndarray,
    sr: int = AUDIO_CONFIG.sample_rate,
    esc50_dir: Optional[Path] = None,
) -> List[Tuple[np.ndarray, str]]:
    """Generuje augmentowane wersje jednej próbki glass_break.

    Cel: z 1 próbki → ~10 wariantów (40 oryginalnych × 10 = 400 augmentowanych).

    Augmentacje:
    1. Time stretch ×0.8 i ×1.2
    2. Pitch shift ±2 semitones
    3. Additive noise (HVAC) przy SNR 10, 5, 0 dB
    4. Volume scaling ×0.5, ×0.7, ×1.5
    5. Room Impulse Response (syntetyczny RIR, RT60=0.3s)

    Args:
        audio: Oryginalne audio glass_break (mono).
        sr: Częstotliwość próbkowania.
        esc50_dir: Ścieżka do ESC-50 (potrzebna do additive noise z klasy air_conditioner).

    Returns:
        Lista krotek (augmented_audio, opis_augmentacji).

    Przykład:
        >>> augmented = augment_glass_break(audio, sr=22050)
        >>> print(f"Wygenerowano {len(augmented)} wariantów")
    """
    augmented: List[Tuple[np.ndarray, str]] = []
    rng = np.random.default_rng(RANDOM_SEED + 1)

    # 1. Time stretch
    for rate in [0.8, 1.2]:
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        augmented.append((stretched, f"time_stretch_{rate}"))

    # 2. Pitch shift
    for semitones in [-2, 2]:
        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semitones)
        augmented.append((shifted, f"pitch_shift_{semitones}st"))

    # 3. Volume scaling
    for scale in [0.5, 0.7, 1.5]:
        scaled = audio * scale
        scaled = np.clip(scaled, -1.0, 1.0)
        augmented.append((scaled, f"volume_{scale}"))

    # 4. Additive noise (HVAC / air_conditioner z ESC-50)
    noise_audio = None
    if esc50_dir is not None:
        # Szukamy plików klasy air_conditioner (klasa 36, fold 1-5)
        ac_files = list((esc50_dir / "audio").glob("*-36-*.wav"))
        if ac_files:
            noise_audio, _ = librosa.load(str(ac_files[0]), sr=sr, mono=True)
    if noise_audio is None:
        # Fallback: biały szum
        noise_audio = rng.normal(0, 0.1, len(audio)).astype(np.float32)

    for snr_db in [10, 5]:
        # Dopasuj długość szumu
        if len(noise_audio) < len(audio):
            noise_rep = np.tile(noise_audio, int(np.ceil(len(audio) / len(noise_audio))))[:len(audio)]
        else:
            noise_rep = noise_audio[:len(audio)]

        # Oblicz SNR i skaluj szum
        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(noise_rep ** 2)
        if noise_power > 0:
            scale = np.sqrt(signal_power / (noise_power * (10 ** (snr_db / 10))))
            noisy = audio + noise_rep * scale
            noisy = np.clip(noisy, -1.0, 1.0).astype(np.float32)
            augmented.append((noisy, f"hvac_noise_snr{snr_db}dB"))

    # 5. Room Impulse Response
    rir = _generate_synthetic_rir(sr, rt60=0.3)
    convolved = fftconvolve(audio, rir, mode='same').astype(np.float32)
    convolved = convolved / (np.abs(convolved).max() + 1e-8)
    augmented.append((convolved, "rir_rt60_0.3s"))

    return augmented


# =============================================================================
# PARSOWANIE ESC-50 METADANYCH
# =============================================================================

def parse_esc50_filename(filename: str) -> Dict:
    """Parsuje nazwę pliku ESC-50 na składowe.

    Format: {fold}-{clip_id}-{take}-{target}.wav
    Przykład: 1-100032-A-0.wav → fold=1, clip_id=100032, take=A, target=0

    Args:
        filename: Nazwa pliku (bez ścieżki).

    Returns:
        Słownik z kluczami: fold, clip_id, take, target.

    Przykład:
        >>> info = parse_esc50_filename("1-100032-A-0.wav")
        >>> print(info["target"])  # 0
    """
    name = filename.replace(".wav", "")
    parts = name.split("-")
    return {
        "fold": int(parts[0]),
        "clip_id": int(parts[1]),
        "take": parts[2],
        "target": int(parts[3]),
    }


# =============================================================================
# BUDOWANIE DATASETU
# =============================================================================

def build_dataset(
    esc50_dir: Optional[Path] = None,
    encoding: str = "ttfs",
    n_timesteps: int = 100,
    data_limit: Optional[int] = None,
    augment: bool = True,
) -> Dict[str, "GlassBreakDataset"]:
    """Buduje kompletny dataset: train, val, test z augmentacją.

    Podział:
    - Test: 10 oryginalnych glass_break + 100 negative (LOCKED)
    - Val: 10 oryginalnych glass_break + 100 negative
    - Train: reszta + augmented glass_break

    Args:
        esc50_dir: Ścieżka do katalogu ESC-50. Domyślnie: auto-download.
        encoding: Metoda enkodowania spike'ów ("rate" lub "ttfs").
        n_timesteps: Liczba kroków czasowych spike train.
        data_limit: Opcjonalny limit liczby plików do załadowania (dla szybkich testów).
        augment: Czy stosować augmentację (True = tak).

    Returns:
        Słownik {"train": Dataset, "val": Dataset, "test": Dataset}.

    Przykład:
        >>> datasets = build_dataset(encoding="ttfs", n_timesteps=100)
        >>> print(f"Train: {len(datasets['train'])}, Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")
    """
    # 1. Pobierz ESC-50 jeśli trzeba
    if esc50_dir is None:
        esc50_dir = download_esc50()
    audio_dir = esc50_dir / "audio"

    # Zbierz wszystkie pliki i podziel najpierw na klasy
    all_files = sorted(audio_dir.glob("*.wav"))
    print(f"[INFO] Znaleziono {len(all_files)} plików audio w katalogu.")

    # Parsuj metadane
    positives: List[Path] = []   # glass_breaking
    negatives: List[Path] = []   # inne klasy
    for f in all_files:
        info = parse_esc50_filename(f.name)
        if info["target"] == ESC50_GLASS_BREAKING_CLASS:
            positives.append(f)
        else:
            negatives.append(f)

    # FIX: Aplikuj data_limit na poziomie klas (balanced subset)
    if data_limit is not None:
        # data_limit dotyczy sumy plików: najpierw bierzemy ile się da z positives
        pos_limit = min(data_limit // 4, len(positives))
        neg_limit = data_limit - pos_limit
        positives = positives[:pos_limit]
        negatives = negatives[:neg_limit]

    print(f"[INFO] Pozytywne (glass_break): {len(positives)}, Negatywne: {len(negatives)}")

    # 3. Stratified split
    rng = np.random.default_rng(RANDOM_SEED)

    # Pozytywne: 10 test, 10 val, reszta train
    pos_shuffled = list(positives)
    rng.shuffle(pos_shuffled)
    n_pos_test = min(10, len(pos_shuffled) // 3)
    n_pos_val = min(10, len(pos_shuffled) // 3)
    pos_test = pos_shuffled[:n_pos_test]
    pos_val = pos_shuffled[n_pos_test:n_pos_test + n_pos_val]
    pos_train = pos_shuffled[n_pos_test + n_pos_val:]

    # Negatywne: 100 test, 100 val, reszta train
    neg_shuffled = list(negatives)
    rng.shuffle(neg_shuffled)
    n_neg_test = min(100, len(neg_shuffled) // 3)
    n_neg_val = min(100, len(neg_shuffled) // 3)
    neg_test = neg_shuffled[:n_neg_test]
    neg_val = neg_shuffled[n_neg_test:n_neg_test + n_neg_val]
    neg_train = neg_shuffled[n_neg_test + n_neg_val:]

    print(f"[INFO] Split — Train: {len(pos_train)}+ / {len(neg_train)}-, "
          f"Val: {len(pos_val)}+ / {len(neg_val)}-, "
          f"Test: {len(pos_test)}+ / {len(neg_test)}-")

    # 4. Preprocessing — załaduj i przekonwertuj na spike trains
    encoder = TTFSEncoder() if encoding == "ttfs" else RateCodingEncoder()

    def process_files(
        file_list: List[Path],
        label: int,
        do_augment: bool = False,
    ) -> List[Tuple[torch.Tensor, int]]:
        """Przetwarza listę plików audio na spike trains."""
        results = []
        for f in tqdm(file_list, desc=f"Preprocessing (label={label})", leave=False):
            try:
                audio, sr = load_audio(str(f))
                features = extract_features(audio, sr)
                # Używamy średniej energii RMS jako wejścia do enkodera
                energy_mean = features[0]  # (n_frames,)
                spikes = encoder.encode(energy_mean, n_timesteps=n_timesteps)
                results.append((spikes, label))

                # Augmentacja (tylko dla glass_break w zbiorze treningowym)
                if do_augment and label == 1:
                    aug_samples = augment_glass_break(audio, sr, esc50_dir)
                    for aug_audio, aug_desc in aug_samples:
                        aug_features = extract_features(aug_audio, sr)
                        aug_energy = aug_features[0]
                        aug_spikes = encoder.encode(aug_energy, n_timesteps=n_timesteps)
                        results.append((aug_spikes, label))
            except Exception as e:
                print(f"[WARN] Pomijam {f.name}: {e}")
        return results

    print("[INFO] Przetwarzam zbiór treningowy...")
    train_data = (
        process_files(pos_train, label=1, do_augment=augment)
        + process_files(neg_train, label=0)
    )
    print("[INFO] Przetwarzam zbiór walidacyjny...")
    val_data = (
        process_files(pos_val, label=1, do_augment=False)
        + process_files(neg_val, label=0)
    )
    print("[INFO] Przetwarzam zbiór testowy...")
    test_data = (
        process_files(pos_test, label=1, do_augment=False)
        + process_files(neg_test, label=0)
    )

    print(f"[INFO] Datasety gotowe — Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    return {
        "train": GlassBreakDataset(train_data, apply_jitter=True),
        "val": GlassBreakDataset(val_data, apply_jitter=False),
        "test": GlassBreakDataset(test_data, apply_jitter=False),
    }


# =============================================================================
# PyTorch DATASET
# =============================================================================

class GlassBreakDataset(Dataset):
    """PyTorch Dataset dla spike trains glass break detection.

    Opcjonalnie stosuje hardware jitter (±1% na spike times) podczas treningu
    — symuluje timing jitter Arduino.

    Attributes:
        data: Lista krotek (spike_tensor, label).
        apply_jitter: Czy dodawać hardware jitter.
        jitter_pct: Procent jitteru.

    Przykład:
        >>> ds = GlassBreakDataset(data_list, apply_jitter=True)
        >>> spikes, label = ds[0]
        >>> print(f"Spikes shape: {spikes.shape}, Label: {label}")
    """

    def __init__(
        self,
        data: List[Tuple[torch.Tensor, int]],
        apply_jitter: bool = False,
        jitter_pct: float = 1.0,
    ) -> None:
        self.data = data
        self.apply_jitter = apply_jitter
        self.jitter_pct = jitter_pct

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Zwraca spike train i etykietę.

        Args:
            idx: Indeks próbki.

        Returns:
            Krotka (spikes [n_channels, n_timesteps], label [1]).
        """
        spikes, label = self.data[idx]

        if self.apply_jitter and self.jitter_pct > 0:
            # Symulacja timing jitter Arduino (±1%)
            noise = torch.randn_like(spikes) * (self.jitter_pct / 100.0)
            jittered = spikes + noise
            spikes = (jittered > 0.5).float()

        return spikes, torch.tensor([label], dtype=torch.float32)


def get_dataloaders(
    datasets: Dict[str, GlassBreakDataset],
    batch_size: int = TRAIN_CONFIG.batch_size,
) -> Dict[str, DataLoader]:
    """Tworzy DataLoadery z datasets.

    Args:
        datasets: Słownik {"train": Dataset, "val": Dataset, "test": Dataset}.
        batch_size: Rozmiar batcha.

    Returns:
        Słownik {"train": DataLoader, "val": DataLoader, "test": DataLoader}.

    Przykład:
        >>> loaders = get_dataloaders(datasets, batch_size=32)
        >>> for batch_spikes, batch_labels in loaders["train"]:
        ...     print(batch_spikes.shape)
        ...     break
    """
    def collate_fn(batch):
        """Custom collate — pad spike trains do jednakowej długości."""
        spikes_list, labels_list = zip(*batch)
        # Pad do max długości w batchu
        max_channels = max(s.shape[0] for s in spikes_list)
        max_timesteps = max(s.shape[1] for s in spikes_list)
        padded_spikes = torch.zeros(len(batch), max_channels, max_timesteps)
        for i, s in enumerate(spikes_list):
            padded_spikes[i, :s.shape[0], :s.shape[1]] = s
        labels = torch.stack(labels_list)
        return padded_spikes, labels

    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=False,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        ),
    }
