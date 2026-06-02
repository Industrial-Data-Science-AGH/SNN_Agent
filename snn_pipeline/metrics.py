# -*- coding: utf-8 -*-
"""
Metryki — precision, recall, F1, accuracy, confusion matrix, latencja.

Wszystkie metryki zaimplementowane jako czyste funkcje PyTorch (bez sklearn w runtime),
ale sklearn.metrics używamy do pretty-print confusion matrix.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple
from sklearn.metrics import confusion_matrix as sk_confusion_matrix


def precision_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Oblicza precision = TP / (TP + FP).

    Args:
        predictions: Predykcje modelu, kształt (N, 1) lub (N,).
        targets: Etykiety ground truth, kształt (N, 1) lub (N,).
        threshold: Próg decyzyjny (domyślnie 0.5).

    Returns:
        Precision jako float [0, 1].

    Przykład:
        >>> pred = torch.tensor([0.9, 0.1, 0.8, 0.3])
        >>> tgt = torch.tensor([1.0, 0.0, 0.0, 1.0])
        >>> print(f"Precision: {precision_score(pred, tgt):.2f}")
    """
    pred_binary = (predictions.flatten() >= threshold).float()
    tgt_flat = targets.flatten()

    tp = ((pred_binary == 1) & (tgt_flat == 1)).sum().float()
    fp = ((pred_binary == 1) & (tgt_flat == 0)).sum().float()

    if (tp + fp) == 0:
        return 1.0  # Brak positive predictions → precision undefined, zwracamy 1.0
    return (tp / (tp + fp)).item()


def recall_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Oblicza recall = TP / (TP + FN).

    Krytyczna metryka — system bezpieczeństwa nie może pomijać zdarzeń glass break.

    Args:
        predictions: Predykcje modelu.
        targets: Etykiety ground truth.
        threshold: Próg decyzyjny.

    Returns:
        Recall jako float [0, 1].

    Przykład:
        >>> pred = torch.tensor([0.9, 0.1, 0.8, 0.3])
        >>> tgt = torch.tensor([1.0, 0.0, 1.0, 1.0])
        >>> print(f"Recall: {recall_score(pred, tgt):.2f}")  # 0.67
    """
    pred_binary = (predictions.flatten() >= threshold).float()
    tgt_flat = targets.flatten()

    tp = ((pred_binary == 1) & (tgt_flat == 1)).sum().float()
    fn = ((pred_binary == 0) & (tgt_flat == 1)).sum().float()

    if (tp + fn) == 0:
        return 1.0  # Brak pozytywnych próbek → recall undefined
    return (tp / (tp + fn)).item()


def f1_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Oblicza F1 = 2 × (precision × recall) / (precision + recall).

    Args:
        predictions: Predykcje modelu.
        targets: Etykiety ground truth.
        threshold: Próg decyzyjny.

    Returns:
        F1 score jako float [0, 1].

    Przykład:
        >>> f1 = f1_score(torch.tensor([0.8, 0.2]), torch.tensor([1.0, 0.0]))
        >>> print(f"F1: {f1:.2f}")
    """
    p = precision_score(predictions, targets, threshold)
    r = recall_score(predictions, targets, threshold)
    if (p + r) == 0:
        return 0.0
    return 2 * p * r / (p + r)


def accuracy_score(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Oblicza accuracy = (TP + TN) / total.

    Args:
        predictions: Predykcje modelu.
        targets: Etykiety ground truth.
        threshold: Próg decyzyjny.

    Returns:
        Accuracy jako float [0, 1].

    Przykład:
        >>> acc = accuracy_score(torch.tensor([0.9, 0.1, 0.8]), torch.tensor([1.0, 0.0, 1.0]))
    """
    pred_binary = (predictions.flatten() >= threshold).float()
    tgt_flat = targets.flatten()
    correct = (pred_binary == tgt_flat).sum().float()
    return (correct / len(tgt_flat)).item()


def false_negative_rate(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """Oblicza False Negative Rate = FN / (TP + FN) = 1 - recall.

    KRYTYCZNA metryka – system bezpieczeństwa musi minimalizować FNR.

    Args:
        predictions: Predykcje modelu.
        targets: Etykiety ground truth.
        threshold: Próg decyzyjny.

    Returns:
        FNR jako float [0, 1].
    """
    return 1.0 - recall_score(predictions, targets, threshold)


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> np.ndarray:
    """Oblicza confusion matrix (2×2).

    Args:
        predictions: Predykcje modelu.
        targets: Etykiety ground truth.
        threshold: Próg decyzyjny.

    Returns:
        Confusion matrix jako numpy array [[TN, FP], [FN, TP]].

    Przykład:
        >>> cm = compute_confusion_matrix(torch.tensor([0.9, 0.1]), torch.tensor([1.0, 0.0]))
        >>> print(cm)  # [[1, 0], [0, 1]]
    """
    pred_binary = (predictions.flatten() >= threshold).long().numpy()
    tgt_np = targets.flatten().long().numpy()
    return sk_confusion_matrix(tgt_np, pred_binary, labels=[0, 1])


def all_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Oblicza wszystkie metryki naraz.

    Args:
        predictions: Predykcje modelu.
        targets: Etykiety ground truth.
        threshold: Próg decyzyjny.

    Returns:
        Słownik z kluczami: precision, recall, f1, accuracy, fnr.

    Przykład:
        >>> metrics = all_metrics(torch.tensor([0.8, 0.2, 0.7]), torch.tensor([1.0, 0.0, 1.0]))
        >>> print(metrics)
    """
    return {
        "precision": precision_score(predictions, targets, threshold),
        "recall": recall_score(predictions, targets, threshold),
        "f1": f1_score(predictions, targets, threshold),
        "accuracy": accuracy_score(predictions, targets, threshold),
        "fnr": false_negative_rate(predictions, targets, threshold),
    }


def latency_ms(
    spike_train: torch.Tensor,
    dt_ms: float = 1.0,
) -> float:
    """Oblicza latencję: czas od początku sygnału do pierwszego spike'a N3.

    Args:
        spike_train: Tensor spike'ów N3, kształt (timesteps,) lub (batch, timesteps).
        dt_ms: Krok czasowy w milisekundach.

    Returns:
        Latencja w milisekundach. Jeśli brak spike'a → float('inf').

    Przykład:
        >>> spikes = torch.tensor([0, 0, 0, 1, 0, 1, 0])
        >>> print(f"Latencja: {latency_ms(spikes):.1f} ms")  # 3.0
    """
    if spike_train.dim() > 1:
        spike_train = spike_train.flatten()

    spike_indices = torch.nonzero(spike_train, as_tuple=True)[0]
    if len(spike_indices) == 0:
        return float('inf')
    return spike_indices[0].item() * dt_ms
