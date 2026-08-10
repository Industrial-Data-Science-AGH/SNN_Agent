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

from encoder.feature_bank.digital_twin_encoder import (
    CHANNEL_EXTRACTORS,
    encode_wav_file,
)

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
    positive_dir: str = "../snn_input/positive",
    negative_dir: str = "../snn_input/negative",
    output_path: str = "feature_bank.npz",
) -> dict:
    X_pos, y_pos, g_pos, names_pos = _encode_dir(positive_dir, label=1, group_offset=0)
    X_neg, y_neg, g_neg, names_neg = _encode_dir(
        negative_dir, label=0, group_offset=len(names_pos)
    )

    if not X_pos and not X_neg:
        print("[BŁĄD] Nie znaleziono żadnych plików .wav - sprawdź ścieżki.")
        sys.exit(1)

    X = np.array(X_pos + X_neg, dtype=np.float64)
    y = np.array(y_pos + y_neg, dtype=np.int64)
    groups = np.array(g_pos + g_neg, dtype=np.int64)
    file_names = np.array(names_pos + names_neg)

    np.savez(
        output_path,
        X=X,
        y=y,
        groups=groups,
        feature_names=np.array(ALL_CHANNELS),
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


# ============================================================
#  WERYFIKACJA BANKU - odpal PRZED podpięciem GA
# ============================================================
def verify_feature_bank(bank: dict) -> None:
    X, y, groups, names = bank["X"], bank["y"], bank["groups"], bank["feature_names"]

    print("\n=== 1. Sanity strukturalny ===")
    print("kształt X:", X.shape, " NaN:", np.isnan(X).any(), " Inf:", np.isinf(X).any())
    print(
        "balans klas: positive =",
        int((y == 1).sum()),
        " negative =",
        int((y == 0).sum()),
    )
    const_cols = [names[i] for i in range(X.shape[1]) if np.std(X[:, i]) < 1e-9]
    if const_cols:
        print("[UWAGA] stałe kolumny (podejrzenie buga):", const_cols)
    else:
        print("brak stałych kolumn - OK")

    print("\n=== 2. Separowalność per kanał (Cohen's d, |d|>0.2 = coś widać) ===")
    for i, name in enumerate(names):
        a, b = X[y == 1, i], X[y == 0, i]
        pooled_std = np.sqrt((a.var() + b.var()) / 2) + 1e-9
        d = (a.mean() - b.mean()) / pooled_std
        flag = "  <-- ma sygnał" if abs(d) > 0.2 else ""
        print(f"  {name:20s} d={d:+.3f}{flag}")

    print(
        "\n=== 3. Baseline: mała regresja logistyczna, split PO PLIKACH (bez wycieku) ==="
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
    train_idx, test_idx = next(gss.split(X, y, groups))

    scaler = StandardScaler().fit(X[train_idx])
    clf = LogisticRegression(max_iter=1000).fit(
        scaler.transform(X[train_idx]), y[train_idx]
    )
    acc = clf.score(scaler.transform(X[test_idx]), y[test_idx])
    print(f"accuracy na plikach nie widzianych w treningu: {acc:.3f}")
    if acc < 0.6:
        print(
            "[UWAGA] blisko przypadku (0.5) - zanim odpalisz GA, sprawdź dataset/etykiety,"
        )
        print(
            "        bo przeszukiwanie architektury nie naprawi braku sygnału w danych."
        )
    else:
        print("jest sygnał ponad przypadek - bank nadaje się jako wejście dla GA.")


if __name__ == "__main__":
    bank = build_feature_bank()
    verify_feature_bank(bank)
