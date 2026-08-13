"""
feature_bank_stats.py
======================
Statystyki i sanity-check'i dla feature banku (X, y, groups, feature_names)
zapisanego przez build_feature_bank.py.

Celowo zero zależności od enkodera/audio (digital_twin_encoder, scipy, wave) -
dzięki temu można to zaimportować gdziekolwiek: notebook, skrypt GA, CI,
bez ciągnięcia za sobą całego pipeline'u ekstrakcji cech. Wystarczy gotowy
feature_bank.npz.

Użycie:
    from feature_bank_stats import load_feature_bank, full_report
    bank = load_feature_bank("feature_bank.npz")
    full_report(bank)

albo pojedyncze funkcje, np. w pętli GA do szybkiego sprawdzenia kanałów:
    from feature_bank_stats import cohens_d_per_channel
    d = cohens_d_per_channel(bank, verbose=False)
"""

from __future__ import annotations

from typing import Any

import numpy as np


def load_feature_bank(path: str = "feature_bank.npz") -> dict[str, Any]:
    """Wczytuje feature_bank.npz do zwykłego dict-a: X, y, groups, feature_names, file_names."""
    data = np.load(path, allow_pickle=True)
    return {
        "X": data["X"],
        "y": data["y"],
        "groups": data["groups"],
        "feature_names": list(data["feature_names"]),
        "file_names": data["file_names"] if "file_names" in data.files else None,
    }


def structural_sanity(bank: dict, verbose: bool = True) -> dict:
    """Kształt, NaN/Inf, balans klas, stałe kolumny (= podejrzenie buga w ekstrakcji)."""
    X, y, names = bank["X"], bank["y"], bank["feature_names"]

    has_nan = bool(np.isnan(X).any())
    has_inf = bool(np.isinf(X).any())
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    stds = X.std(axis=0)
    constant_columns = [names[i] for i in range(X.shape[1]) if stds[i] < 1e-9]

    result = {
        "shape": X.shape,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "constant_columns": constant_columns,
    }

    if verbose:
        print("=== Sanity strukturalny ===")
        print("kształt X:", X.shape, " NaN:", has_nan, " Inf:", has_inf)
        print("balans klas: positive =", n_pos, " negative =", n_neg)
        if constant_columns:
            print("[UWAGA] stałe kolumny (podejrzenie buga):", constant_columns)
        else:
            print("brak stałych kolumn - OK")

    return result


def cohens_d_per_channel(bank: dict, verbose: bool = True) -> dict[str, float]:
    """Cohen's d (positive vs negative) dla każdego kanału - szybki sygnał, które
    kanały w ogóle niosą informację, zanim odpalisz GA."""
    X, y, names = bank["X"], bank["y"], bank["feature_names"]
    result: dict[str, float] = {}
    for i, name in enumerate(names):
        a, b = X[y == 1, i], X[y == 0, i]
        pooled_std = np.sqrt((a.var() + b.var()) / 2) + 1e-9
        result[name] = float((a.mean() - b.mean()) / pooled_std)

    if verbose:
        print("\n=== Separowalność per kanał (Cohen's d, |d|>0.2 = coś widać) ===")
        for name, d in result.items():
            flag = "  <-- ma sygnał" if abs(d) > 0.2 else ""
            print(f"  {name:20s} d={d:+.3f}{flag}")

    return result


def baseline_classifier_report(
    bank: dict,
    test_size: float = 0.3,
    random_state: int = 0,
    verbose: bool = True,
) -> dict:
    """Mała regresja logistyczna, split PO PLIKACH (groups) - bez wycieku ramek
    z tego samego nagrania między train/test.

    Zwraca kilka metryk, nie tylko accuracy: przy niezbalansowanych klasach (jak
    w Waszym banku, ~2.1:1) accuracy jest myląca - to właśnie balanced_accuracy /
    f1 / roc_auc powinny sterować fitness w GA, zgodnie z wnioskiem z README."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    X, y, groups = bank["X"], bank["y"], bank["groups"]

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))

    scaler = StandardScaler().fit(X[train_idx])
    clf = LogisticRegression(max_iter=1000).fit(
        scaler.transform(X[train_idx]), y[train_idx]
    )

    y_true = y[test_idx]
    y_pred = clf.predict(scaler.transform(X[test_idx]))
    y_score = clf.predict_proba(scaler.transform(X[test_idx]))[:, 1]
    majority_baseline = float(max(y_true.mean(), 1 - y_true.mean()))

    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "majority_class_baseline": majority_baseline,
    }

    if verbose:
        print("\n=== Baseline: regresja logistyczna, split PO PLIKACH ===")
        print(
            f"accuracy:            {result['accuracy']:.3f}  (naiwny baseline klasy większościowej: {majority_baseline:.3f})"
        )
        print(f"balanced_accuracy:   {result['balanced_accuracy']:.3f}")
        print(f"f1:                  {result['f1']:.3f}")
        print(f"roc_auc:             {result['roc_auc']:.3f}")
        if result["balanced_accuracy"] < 0.6:
            print(
                "[UWAGA] blisko przypadku - zanim odpalisz GA, sprawdź dataset/etykiety."
            )
        else:
            print("jest sygnał ponad przypadek - bank nadaje się jako wejście dla GA.")

    return result


def full_report(bank: dict, verbose: bool = True) -> dict:
    """Komplet: sanity + Cohen's d + baseline. Odpowiednik dawnego verify_feature_bank(),
    ale zwraca ustrukturyzowany wynik (do zapisania/logowania), nie tylko printuje."""
    return {
        "structural": structural_sanity(bank, verbose=verbose),
        "cohens_d": cohens_d_per_channel(bank, verbose=verbose),
        "baseline": baseline_classifier_report(bank, verbose=verbose),
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "feature_bank.npz"
    full_report(load_feature_bank(path))
