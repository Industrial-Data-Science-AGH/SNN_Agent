"""
build_feature_bank.py
======================
Buduje "feature bank" z datasetu snn_input/positive i snn_input/negative.

"""

from __future__ import annotations

import glob
import os
import sys
import numpy as np
from digital_twin_encoder import CHANNEL_EXTRACTORS, encode_wav_file


# Dodanie katalogu skryptu do sys.path (rozwiazuje problemy z importami lokalnymi)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

ALL_CHANNELS = list(CHANNEL_EXTRACTORS.keys())


def _encode_dir(input_dir: str, label: int, group_offset: int):
    wav_files = sorted(glob.glob(os.path.join(input_dir, "*.wav")))
    if not wav_files:
        print(f"[UWAGA] brak plików .wav w '{input_dir}'")
        return [], [], [], []

    X_rows, y_rows, group_rows, file_names = [], [], [], []
    n_files = len(wav_files)

    for i, wav_path in enumerate(wav_files):
        gid = group_offset + i

        if i % 200 == 0 or i == n_files - 1:
            print(f"  [{input_dir}] {i + 1}/{n_files} plików...")

        rows = encode_wav_file(wav_path, channels=ALL_CHANNELS)
        rows = [r for r in rows if not r.get("_priming")]

        for r in rows:
            X_rows.append([r[c] for c in ALL_CHANNELS])
            y_rows.append(label)
            group_rows.append(gid)

        file_names.append(os.path.basename(wav_path))

    return X_rows, y_rows, group_rows, file_names


def build_feature_bank(
    positive_dir: str = "snn_input/positive",
    negative_dir: str = "snn_input/negative",
    output_path: str = "feature_bank.npz",
) -> dict:
    """Encode positive/negative WAV directories and save feature_bank.npz."""
    X_pos, y_pos, g_pos, names_pos = _encode_dir(positive_dir, label=1, group_offset=0)
    X_neg, y_neg, g_neg, names_neg = _encode_dir(
        negative_dir, label=0, group_offset=len(names_pos)
    )

    if not X_pos and not X_neg:
        print("[BŁĄD] Nie znaleziono żadnych plików .wav - sprawdź ścieżki.")
        sys.exit(1)

    X = np.asarray(X_pos + X_neg, dtype=np.float64)
    y = np.asarray(y_pos + y_neg, dtype=np.int64)
    groups = np.asarray(g_pos + g_neg, dtype=np.int64)
    file_names = np.asarray(names_pos + names_neg)

    np.savez(
        output_path,
        X=X,
        y=y,
        groups=groups,
        feature_names=np.asarray(ALL_CHANNELS),
        file_names=file_names,
    )

    print(f"[OK] Zapisano feature bank -> {output_path}")
    print(f"     ramki: {X.shape[0]}, kanały: {X.shape[1]}, plików: {len(file_names)}")

    return {
        "X": X,
        "y": y,
        "groups": groups,
        "feature_names": ALL_CHANNELS,
        "file_names": file_names,
    }


if __name__ == "__main__":
    from feature_bank_stats import full_report

    bank = build_feature_bank()
    full_report(bank)
