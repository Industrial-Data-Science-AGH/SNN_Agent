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


def accuracy_drop(
    baseline_accuracy: float,
    trained_accuracy: float,
) -> float:
    """Calculates accuracy drop between baseline and trained model.

    Measures how much accuracy was lost during training (quantization, mismatch, etc).
    Negative values indicate accuracy improvement.

    Args:
        baseline_accuracy: Accuracy of baseline model (before training).
        trained_accuracy: Accuracy of trained model (after HAT/QAT).

    Returns:
        Accuracy drop as percentage points. Negative = improvement, Positive = degradation.
        Example: drop=-0.05 means 5% accuracy gain. drop=0.03 means 3% accuracy loss.

    Przykład:
        >>> drop = accuracy_drop(baseline_accuracy=0.92, trained_accuracy=0.89)
        >>> print(f"Accuracy drop: {drop:.2%}")  # 3.00%
    """
    return baseline_accuracy - trained_accuracy


def thermal_resistance_score(
    nominal_recall: float,
    worst_case_recall: float,
    thermal_drift_pct: float = 2.0,
) -> Dict[str, float]:
    """Quantifies system robustness to thermal drift.

    Thermal resistance is the ability to maintain performance under thermal variations.
    Higher thermal_robustness (closer to 1.0) means better resistance to thermal effects.

    Args:
        nominal_recall: Recall at nominal conditions (25°C).
        worst_case_recall: Recall under worst-case thermal drift.
        thermal_drift_pct: Percentage drift applied (for reference, default 2%).

    Returns:
        Dict with:
            - robustness_score: [0, 1], higher = more robust (1.0 = no degradation)
            - recall_change: absolute change in recall
            - recall_change_pct: relative change as percentage
            - degradation_severity: "low" (<1%), "medium" (1-5%), "high" (>5%)

    Przykład:
        >>> thermal = thermal_resistance_score(nominal_recall=0.92, worst_case_recall=0.87)
        >>> print(f"Thermal robustness: {thermal['robustness_score']:.2%}")
        >>> print(f"Degradation: {thermal['degradation_severity']}")
    """
    if nominal_recall == 0:
        return {
            "robustness_score": 0.0,
            "recall_change": 0.0,
            "recall_change_pct": 0.0,
            "degradation_severity": "undefined",
        }

    recall_change = worst_case_recall - nominal_recall
    recall_change_pct = abs(recall_change / nominal_recall)

    # Robustness score: 1.0 = no degradation, 0.0 = total failure
    # Map recall_change_pct to robustness: 0% change → 1.0, 50% change → 0.5, 100% change → 0.0
    robustness_score = max(0.0, 1.0 - recall_change_pct)

    # Classify severity
    if recall_change_pct < 0.01:
        severity = "low"
    elif recall_change_pct < 0.05:
        severity = "medium"
    else:
        severity = "high"

    return {
        "robustness_score": robustness_score,
        "recall_change": recall_change,
        "recall_change_pct": recall_change_pct,
        "degradation_severity": severity,
    }


def energy_efficiency(
    spike_rate: float,
    vcc: float = 5.0,
    max_current_ua: float = 100.0,
    inference_duration_ms: float = 100.0,
) -> Dict[str, float]:
    """Calculates energy efficiency metrics for the SNN inference.

    Energy efficiency estimates power consumption based on spike activity.
    Lower energy per correct detection = better efficiency.

    Args:
        spike_rate: Average spikes per neuron per timestep [0, 1] (fraction of active neurons).
        vcc: Supply voltage in Volts (default 5V).
        max_current_ua: Maximum current per active synapse in µA (default 100µA).
        inference_duration_ms: Duration of inference window in ms (default 100ms).

    Returns:
        Dict with:
            - avg_power_mw: Average power consumption in milliwatts
            - energy_per_inference_uj: Energy per inference in microjoules
            - efficiency_score: [0, 1] where 1.0 = lowest possible power

    Beispiel:
        >>> eff = energy_efficiency(spike_rate=0.15, vcc=5.0, max_current_ua=100.0)
        >>> print(f"Power: {eff['avg_power_mw']:.2f} mW")
        >>> print(f"Energy/inference: {eff['energy_per_inference_uj']:.2f} µJ")
    """
    # Assume 4 neurons active (N1, N2, N3, N_inh)
    n_neurons = 4

    # Current is proportional to spike rate:
    # current_ua = spike_rate * n_neurons * max_current_per_neuron
    avg_current_ua = spike_rate * n_neurons * (max_current_ua / n_neurons)

    # Power = V * I (convert to mW: V in volts, I in mA = µA/1000)
    avg_power_ua = avg_current_ua  # in µA
    avg_power_ma = avg_power_ua / 1000.0  # convert to mA
    avg_power_mw = vcc * avg_power_ma  # Power in mW

    # Energy per inference = Power * Duration
    # Convert ms to seconds: duration_s = inference_duration_ms / 1000
    energy_mj = avg_power_mw * (inference_duration_ms / 1000.0)  # in mJ
    energy_uj = energy_mj * 1000.0  # convert to µJ

    # Efficiency score: 1.0 at spike_rate=0 (no power), scales down as spike_rate increases
    # At spike_rate=0.3 (30% active), efficiency = 0.7
    efficiency_score = max(0.0, 1.0 - spike_rate)

    return {
        "avg_power_mw": avg_power_mw,
        "energy_per_inference_uj": energy_uj,
        "efficiency_score": efficiency_score,
        "spike_rate": spike_rate,
        "inference_duration_ms": inference_duration_ms,
    }


def latency_to_detection(
    all_latencies: list,
    valid_only: bool = True,
) -> Dict[str, float]:
    """Calculates latency-to-detection statistics.

    Latency is critical for security: faster detection = faster response to glass break.

    Args:
        all_latencies: List of latencies in milliseconds for all test samples.
        valid_only: If True, ignore inf latencies (missed detections); if False, treat inf as worst-case.

    Returns:
        Dict with:
            - mean_latency_ms: Average detection latency
            - min_latency_ms: Fastest detection
            - max_latency_ms: Slowest detection (excluding inf if valid_only=True)
            - std_latency_ms: Standard deviation of latencies
            - p95_latency_ms: 95th percentile latency (5% of detections slower)
            - p99_latency_ms: 99th percentile latency (1% of detections slower)

    Beispiel:
        >>> latencies = [2.5, 3.0, 2.8, 4.1, 3.2, 15.0]
        >>> stats = latency_to_detection(latencies)
        >>> print(f"Mean latency: {stats['mean_latency_ms']:.1f} ms")
        >>> print(f"95th percentile: {stats['p95_latency_ms']:.1f} ms")
    """
    if not all_latencies:
        return {
            "mean_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "std_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "p99_latency_ms": 0.0,
        }

    # Filter out infinite latencies if requested
    latencies_array = np.array(all_latencies)
    if valid_only:
        latencies_array = latencies_array[np.isfinite(latencies_array)]

    if len(latencies_array) == 0:
        return {
            "mean_latency_ms": float('inf'),
            "min_latency_ms": float('inf'),
            "max_latency_ms": float('inf'),
            "std_latency_ms": float('inf'),
            "p95_latency_ms": float('inf'),
            "p99_latency_ms": float('inf'),
        }

    return {
        "mean_latency_ms": float(np.mean(latencies_array)),
        "min_latency_ms": float(np.min(latencies_array)),
        "max_latency_ms": float(np.max(latencies_array)),
        "std_latency_ms": float(np.std(latencies_array)),
        "p95_latency_ms": float(np.percentile(latencies_array, 95)),
        "p99_latency_ms": float(np.percentile(latencies_array, 99)),
    }
