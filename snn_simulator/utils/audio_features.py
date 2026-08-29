# --- audio_features.py ---
# Ekstrakcja cech audio: STFT, pasma, envelope
import numpy as np
import librosa
import torch


def features_to_spikes(features):
    return torch.tensor(features).float()


def extract_features(path, sr=16000, T=100):
    y, sr = librosa.load(path, sr=sr)

    # --- 1. STFT ---
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=256))

    freqs = librosa.fft_frequencies(sr=sr)

    # --- 2. pasma ---
    hf_mask = freqs > 2000
    lf_mask = freqs < 500

    hf_energy = S[hf_mask].mean(axis=0)
    lf_energy = S[lf_mask].mean(axis=0)

    # --- 3. envelope ---
    envelope = librosa.onset.onset_strength(y=y, sr=sr)

    # --- 4. wyrównanie długości ---
    min_len = min(len(hf_energy), len(lf_energy), len(envelope))

    hf_energy = hf_energy[:min_len]
    lf_energy = lf_energy[:min_len]
    envelope = envelope[:min_len]

    # --- 5. normalizacja ---
    hf_energy = hf_energy / (hf_energy.max() + 1e-6)
    lf_energy = lf_energy / (lf_energy.max() + 1e-6)
    envelope = envelope / (envelope.max() + 1e-6)

    # --- 6. stack ---
    features = np.stack([hf_energy, lf_energy, envelope], axis=1)

    # --- 7. resize do T ---
    idx = np.linspace(0, len(features) - 1, T).astype(int)
    features = features[idx]

    return features  # (T, 3)
