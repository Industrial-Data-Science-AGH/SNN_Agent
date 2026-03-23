# -*- coding: utf-8 -*-
"""
Analiza wrażliwości — testuje jak degraduje się recall gdy każda waga
zostanie zaokrąglona do E24. Generuje heatmapę wrażliwości i identyfikuje
synapsy wymagające potencjometrów cyfrowych (MCP4151).
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

from snn_pipeline.config import DEVICE, PATH_CONFIG
from snn_pipeline.e24_quantizer import quantize_to_e24
from snn_pipeline.metrics import recall_score
from snn_pipeline.snn_model import GlassBreakSNN


def sensitivity_analysis(
    model: GlassBreakSNN,
    test_loader: DataLoader,
    perturbation_pct: float = 1.0,
    n_perturbations: int = 20,
) -> Dict[str, float]:
    """Analiza wrażliwości — dla każdej wagi mierzy degradację recall.

    Dla każdej wagi synaptycznej:
    1. Zapisz oryginalną wartość
    2. Zaokrąglij do najbliższej E24
    3. Dodaj ±perturbation_pct szum
    4. Zmierz recall na test set
    5. Oblicz max degradację recall

    Args:
        model: Wytrenowany model SNN.
        test_loader: DataLoader zbioru testowego.
        perturbation_pct: Procent perturbacji wagi.
        n_perturbations: Liczba losowych perturbacji per waga.

    Returns:
        Słownik {nazwa_wagi: max_recall_degradation_percent}.

    Przykład:
        >>> sens = sensitivity_analysis(model, test_loader)
        >>> for name, deg in sorted(sens.items(), key=lambda x: -x[1]):
        ...     print(f"{name}: {deg:.1f}% degradacji recall")
    """
    model.eval()
    model.set_quantize_mode("none")
    model.enable_mismatch(False)

    # Baseline recall (oryginalne wagi)
    baseline_recall = _evaluate_recall(model, test_loader)
    print(f"[Sensitivity] Baseline recall: {baseline_recall:.2f}")

    weight_params = {
        "w_n1": model.w_n1,
        "w_n2": model.w_n2,
        "w_n3_from_n1": model.w_n3_from_n1,
        "w_n3_from_n2": model.w_n3_from_n2,
        "w_inh": model.w_inh,
        "w_inh_to_n3": model.w_inh_to_n3,
    }

    sensitivities: Dict[str, float] = {}

    for name, param in tqdm(weight_params.items(), desc="Sensitivity analysis"):
        original_val = param.data.clone()
        worst_degradation = 0.0

        for i in range(n_perturbations):
            # Perturbacja: zaokrąglenie E24 + losowy szum ±pct%
            noise_factor = 1.0 + (np.random.randn() * perturbation_pct / 100.0)

            if name == "w_inh_to_n3":
                # Waga hamująca — ujemna, kwantyzujemy wartość bezwzgl.
                perturbed = original_val * noise_factor
            else:
                # Kwantyzacja E24 + szum
                quantized = quantize_to_e24(original_val.abs())
                perturbed = quantized * noise_factor
                if original_val.item() < 0:
                    perturbed = -perturbed

            param.data.copy_(perturbed)
            perturbed_recall = _evaluate_recall(model, test_loader)
            degradation = max(0, baseline_recall - perturbed_recall)
            worst_degradation = max(worst_degradation, degradation)

        # Przywróć oryginalną wartość
        param.data.copy_(original_val)
        sensitivities[name] = worst_degradation * 100  # w procentach

    print(f"\n[Sensitivity] Wyniki (max degradacja recall %):")
    for name, deg in sorted(sensitivities.items(), key=lambda x: -x[1]):
        flag = " ⚠️ KRYTYCZNA → MCP4151" if deg > 5.0 else ""
        print(f"  {name:<18}: {deg:.1f}%{flag}")

    return sensitivities


def generate_heatmap(
    sensitivities: Dict[str, float],
    save_path: Optional[str] = None,
) -> None:
    """Generuje heatmapę wrażliwości wag.

    Args:
        sensitivities: Słownik {nazwa_wagi: degradacja_recall_%}.
        save_path: Ścieżka do zapisu. Domyślnie: output/sensitivity_heatmap.png.

    Przykład:
        >>> generate_heatmap({"w_n1": 2.3, "w_n2": 7.8, "w_n3_from_n1": 12.1})
    """
    if save_path is None:
        save_path = str(PATH_CONFIG.output_dir / "sensitivity_heatmap.png")

    names = list(sensitivities.keys())
    values = [sensitivities[n] for n in names]

    fig, ax = plt.subplots(figsize=(10, 4))

    # Sortuj od najbardziej wrażliwych
    sorted_indices = np.argsort(values)[::-1]
    sorted_names = [names[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    # Kolorowa mapa: zielony → żółty → czerwony
    colors = []
    for v in sorted_values:
        if v < 2:
            colors.append("#28a745")  # zielony — bezpieczne
        elif v < 5:
            colors.append("#ffc107")  # żółty — uwaga
        else:
            colors.append("#dc3545")  # czerwony — krytyczne

    bars = ax.barh(sorted_names, sorted_values, color=colors, edgecolor="black", linewidth=0.5)

    # Linia progowa 5%
    ax.axvline(x=5.0, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="Próg MCP4151 (5%)")

    ax.set_xlabel("Max degradacja recall (%)", fontsize=11)
    ax.set_title("Analiza wrażliwości wag synaptycznych na kwantyzację E24", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

    # Adnotacje na barach
    for bar, val in zip(bars, sorted_values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Sensitivity heatmap saved to {save_path}")


def identify_mcp4151_candidates(
    sensitivities: Dict[str, float],
    threshold_pct: float = 5.0,
) -> List[str]:
    """Identyfikuje synapsy wymagające potencjometrów cyfrowych.

    Synapsy z wrażliwością > threshold_pct% recall na ±1% wagi
    powinny być implementowane jako MCP4151 zamiast stałych rezystorów.

    Args:
        sensitivities: Wynik sensitivity_analysis().
        threshold_pct: Próg wrażliwości (domyślnie 5%).

    Returns:
        Lista nazw synaps wymagających MCP4151.

    Przykład:
        >>> candidates = identify_mcp4151_candidates(sens)
        >>> print(f"Potrzeba {len(candidates)} potencjometrów cyfrowych")
    """
    candidates = [name for name, deg in sensitivities.items() if deg > threshold_pct]
    if candidates:
        print(f"[MCP4151] {len(candidates)} synaps wymaga potencjometrów: {candidates}")
    else:
        print("[MCP4151] Żadna synapsa nie wymaga potencjometru cyfrowego (wszystkie <5%).")
    return candidates


@torch.no_grad()
def _evaluate_recall(model: GlassBreakSNN, loader: DataLoader) -> float:
    """Pomocnicza: oblicza recall modelu na zbiorze danych.

    Args:
        model: Model SNN.
        loader: DataLoader.

    Returns:
        Recall jako float.
    """
    model.eval()
    all_preds = []
    all_targets = []

    for spikes, labels in loader:
        spikes = spikes.to(DEVICE)
        labels = labels.to(DEVICE)
        trigger, _ = model(spikes)
        all_preds.append(trigger)
        all_targets.append(labels)

    if not all_preds:
        return 0.0

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return recall_score(all_preds, all_targets)
