# -*- coding: utf-8 -*-
"""
Ewaluacja pełna — benchmark porównawczy, thermal drift simulation,
analiza błędów (FN/FP), weight error distribution, power estimate.

Generuje raport końcowy z tabelą: Baseline vs HAT vs HAT+QAT.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from snn_pipeline.config import (
    BASELINE_NEURONS,
    DEVICE,
    HW_CONFIG,
    PATH_CONFIG,
    TRAIN_CONFIG,
)
from snn_pipeline.e24_quantizer import (
    quantize_to_e24,
    quantize_to_e24_with_error,
    weight_to_resistance,
)
from snn_pipeline.metrics import (
    all_metrics,
    compute_confusion_matrix,
    latency_ms,
    precision_score,
    recall_score,
)
from snn_pipeline.snn_model import GlassBreakSNN


@torch.no_grad()
def evaluate_model(
    model: GlassBreakSNN,
    test_loader: DataLoader,
    label: str = "Model",
) -> Dict[str, Any]:
    """Pełna ewaluacja modelu na zbiorze testowym.

    Args:
        model: Model SNN.
        test_loader: DataLoader testowy.
        label: Nazwa modelu do raportowania.

    Returns:
        Słownik z metrykami: precision, recall, f1, accuracy, fnr, confusion_matrix,
        avg_latency_ms, weights, thresholds.

    Przykład:
        >>> results = evaluate_model(model, test_loader, label="HAT")
        >>> print(f"Recall: {results['recall']:.2f}")
    """
    model.eval()
    model.enable_mismatch(False)

    all_preds = []
    all_targets = []
    all_latencies = []

    for batch_spikes, batch_labels in test_loader:
        batch_spikes = batch_spikes.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)
        trigger, neuron_spikes = model(batch_spikes)
        all_preds.append(trigger.cpu())
        all_targets.append(batch_labels.cpu())

        # Latencja per sample (czas do pierwszego spike'a N3)
        for i in range(trigger.shape[0]):
            lat = latency_ms(neuron_spikes["N3"][i].cpu())
            all_latencies.append(lat)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    metrics = all_metrics(all_preds, all_targets)
    cm = compute_confusion_matrix(all_preds, all_targets)

    # Średnia latencja (tylko dla poprawnych detekcji)
    valid_latencies = [l for l in all_latencies if l < float('inf')]
    avg_latency = np.mean(valid_latencies) if valid_latencies else float('inf')

    result = {
        **metrics,
        "confusion_matrix": cm,
        "avg_latency_ms": avg_latency,
        "weights": model.get_weights_dict(),
        "thresholds": model.get_thresholds_dict(),
        "label": label,
    }

    print(f"\n{'=' * 50}")
    print(f"  EWALUACJA: {label}")
    print(f"{'=' * 50}")
    print(f"  Precision:    {metrics['precision']:.4f}")
    print(f"  Recall:       {metrics['recall']:.4f}")
    print(f"  F1:           {metrics['f1']:.4f}")
    print(f"  Accuracy:     {metrics['accuracy']:.4f}")
    print(f"  FNR:          {metrics['fnr']:.4f}")
    print(f"  Avg latency:  {avg_latency:.1f} ms")
    print(f"  Confusion matrix:")
    print(f"    TN={cm[0,0]:4d}  FP={cm[0,1]:4d}")
    print(f"    FN={cm[1,0]:4d}  TP={cm[1,1]:4d}")
    print(f"{'=' * 50}")

    return result


@torch.no_grad()
def thermal_drift_simulation(
    model: GlassBreakSNN,
    test_loader: DataLoader,
    n_runs: int = 100,
    drift_pct: float = 2.0,
) -> Dict[str, float]:
    """Symulacja dryfu termicznego — uruchamia ewaluację n_runs razy
    z losowym szumem ±drift_pct% na wagach.

    Raportuje:
    - mean recall ± std recall
    - worst-case recall (5th percentile)
    - probability(recall < 0.80)

    Args:
        model: Model SNN.
        test_loader: DataLoader testowy.
        n_runs: Liczba powtórzeń symulacji.
        drift_pct: Procent dryfu (domyślnie ±2%).

    Returns:
        Słownik z wynikami symulacji.

    Przykład:
        >>> drift = thermal_drift_simulation(model, test_loader)
        >>> print(f"Mean recall: {drift['mean_recall']:.2f} ± {drift['std_recall']:.2f}")
    """
    model.eval()

    recalls = []
    weight_params = [
        model.w_n1, model.w_n2, model.w_n3_from_n1,
        model.w_n3_from_n2, model.w_inh, model.w_inh_to_n3,
    ]

    # Zapisz oryginalne wagi
    original_weights = [p.data.clone() for p in weight_params]

    print(f"[Thermal Drift] Symulacja {n_runs} uruchomień z ±{drift_pct}% szumem...")

    for run in tqdm(range(n_runs), desc="Thermal drift sim"):
        # Dodaj losowy szum do wag
        for param, orig in zip(weight_params, original_weights):
            noise = torch.randn_like(orig) * (drift_pct / 100.0)
            param.data.copy_(orig * (1.0 + noise))

        # Zbierz predykcje
        run_preds = []
        run_targets = []
        for spikes, labels in test_loader:
            spikes = spikes.to(DEVICE)
            trigger, _ = model(spikes)
            run_preds.append(trigger.cpu())
            run_targets.append(labels)

        run_preds = torch.cat(run_preds)
        run_targets = torch.cat(run_targets)
        r = recall_score(run_preds, run_targets)
        recalls.append(r)

    # Przywróć oryginalne wagi
    for param, orig in zip(weight_params, original_weights):
        param.data.copy_(orig)

    recalls_np = np.array(recalls)
    result = {
        "mean_recall": float(np.mean(recalls_np)),
        "std_recall": float(np.std(recalls_np)),
        "worst_case_recall_5pct": float(np.percentile(recalls_np, 5)),
        "prob_recall_below_80": float(np.mean(recalls_np < 0.80)),
        "min_recall": float(np.min(recalls_np)),
        "max_recall": float(np.max(recalls_np)),
    }

    print(f"\n[Thermal Drift] Wyniki ({n_runs} runs, ±{drift_pct}% szum):")
    print(f"  Mean recall:     {result['mean_recall']:.4f} ± {result['std_recall']:.4f}")
    print(f"  Worst-case (5%): {result['worst_case_recall_5pct']:.4f}")
    print(f"  P(recall < 0.80): {result['prob_recall_below_80']:.2%}")
    print(f"  Min/Max recall:  [{result['min_recall']:.4f}, {result['max_recall']:.4f}]")

    return result


def weight_error_histogram(
    model: GlassBreakSNN,
    save_path: Optional[str] = None,
) -> Dict[str, Tuple[float, float]]:
    """Histogram błędów kwantyzacji |w_quantized - w_continuous| per neuron.

    Args:
        model: Model SNN.
        save_path: Ścieżka do zapisu wykresu.

    Returns:
        Słownik {nazwa: (error_abs, error_pct)}.

    Przykład:
        >>> errors = weight_error_histogram(model)
    """
    if save_path is None:
        save_path = str(PATH_CONFIG.output_dir / "weight_error_histogram.png")

    weights = model.get_weights_dict()
    errors: Dict[str, Tuple[float, float]] = {}

    fig, ax = plt.subplots(figsize=(10, 5))

    names = []
    abs_errors = []
    pct_errors = []

    for name, val in weights.items():
        w = torch.tensor([abs(val)])
        w_q, err_abs, err_pct = quantize_to_e24_with_error(w)
        errors[name] = (err_abs.item(), err_pct.item())
        names.append(name)
        abs_errors.append(err_abs.item())
        pct_errors.append(err_pct.item())

    x = np.arange(len(names))
    bars = ax.bar(x, pct_errors, color=["#dc3545" if p > 5 else "#28a745" for p in pct_errors],
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Błąd kwantyzacji (%)")
    ax.set_title("Błąd E24 kwantyzacji per waga synaptyczna", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=5.0, color="red", linestyle="--", alpha=0.5, label="Próg 5%")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Weight error histogram saved to {save_path}")

    return errors


def power_estimate(
    model: GlassBreakSNN,
    spike_rate: float = 50.0,
    spike_width_s: float = 0.001,
    vcc: float = HW_CONFIG.vcc,
) -> Dict[str, float]:
    """Szacuje pobór mocy: P = Σ V²/R_syn × duty_cycle.

    Args:
        model: Model SNN.
        spike_rate: Średnia częstotliwość spike'ów (Hz).
        spike_width_s: Szerokość spike'a (sekundy).
        vcc: Napięcie zasilania (V).

    Returns:
        Słownik z mocami per synapsa i łączną.

    Przykład:
        >>> power = power_estimate(model)
        >>> print(f"Łączna moc: {power['total_mW']:.3f} mW")
    """
    duty_cycle = spike_rate * spike_width_s

    weights = model.get_weights_dict()
    power_per_synapse: Dict[str, float] = {}
    total_power_w = 0.0

    for name, w_val in weights.items():
        r_syn = weight_to_resistance(abs(w_val))
        if r_syn > 0:
            p = (vcc ** 2 / r_syn) * duty_cycle  # Watty
        else:
            p = 0.0
        power_per_synapse[name] = p * 1000  # mW
        total_power_w += p

    power_per_synapse["total_mW"] = total_power_w * 1000
    power_per_synapse["total_uW"] = total_power_w * 1_000_000

    print(f"\n[Power Estimate] (spike_rate={spike_rate}Hz, duty_cycle={duty_cycle:.4f})")
    for name, p_mw in power_per_synapse.items():
        if name.startswith("w_"):
            print(f"  {name:<18}: {p_mw:.4f} mW")
    print(f"  {'ŁĄCZNIE':<18}: {power_per_synapse['total_mW']:.4f} mW ({power_per_synapse['total_uW']:.1f} µW)")

    return power_per_synapse


def benchmark_table(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None,
) -> str:
    """Generuje tabelę porównawczą Baseline vs HAT vs HAT+QAT.

    Args:
        results: Lista wyników z evaluate_model() dla każdego modelu.
        save_path: Ścieżka do zapisu tabeli (tekst).

    Returns:
        Sformatowana tabela jako string.

    Przykład:
        >>> table = benchmark_table([baseline_res, hat_res, qat_res])
        >>> print(table)
    """
    if save_path is None:
        save_path = str(PATH_CONFIG.output_dir / "benchmark_table.txt")

    header = f"{'Model':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Accuracy':>10} {'FNR':>10} {'Latency':>10} {'HW feasible':>12}"
    sep = "-" * len(header)

    lines = [
        "=" * len(header),
        "  BENCHMARK PORÓWNAWCZY — Baseline vs HAT vs HAT+QAT",
        "=" * len(header),
        header,
        sep,
    ]

    for r in results:
        label = r.get("label", "—")
        # Sprawdź czy wagi są w zakresie E24
        hw_ok = "✓" if all(abs(v) <= 0.95 for v in r.get("weights", {}).values()) else "?"
        lines.append(
            f"{label:<20} {r['precision']:>10.4f} {r['recall']:>10.4f} "
            f"{r['f1']:>10.4f} {r['accuracy']:>10.4f} {r['fnr']:>10.4f} "
            f"{r.get('avg_latency_ms', float('inf')):>9.1f}ms {hw_ok:>12}"
        )

    lines.append(sep)
    table_str = "\n".join(lines)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(table_str)
    print(f"\n{table_str}")
    print(f"\n[INFO] Benchmark table saved to {save_path}")

    return table_str
