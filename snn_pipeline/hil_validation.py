# -*- coding: utf-8 -*-
"""
Hardware-in-the-Loop Validation — symulacja ±1% szumu rezystorów,
identyfikacja krytycznych synaps, tabela komend SPI dla MCP4151.

Generuje raport: "z prawdopodobieństwem X% recall będzie ≥ Y%"
i tabelę programowania potencjometrów cyfrowych.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from snn_pipeline.config import DEVICE, HW_CONFIG, PATH_CONFIG
from snn_pipeline.e24_quantizer import weight_to_resistance, weight_to_nearest_e24_resistance
from snn_pipeline.metrics import recall_score
from snn_pipeline.snn_model import GlassBreakSNN


@torch.no_grad()
def hil_simulation(
    model: GlassBreakSNN,
    test_loader: DataLoader,
    n_scenarios: int = 100,
    noise_pct: float = 1.0,
) -> Dict[str, Any]:
    """Symulacja Hardware-in-the-Loop: 100 scenariuszy z ±1% szumem rezystorów.

    Wczytuje wyeksportowane wagi, dodaje losowy szum (symulacja tolerancji
    rezystorów 1%), mierzy recall i generuje raport statystyczny.

    Args:
        model: Wytrenowany model SNN.
        test_loader: DataLoader testowy.
        n_scenarios: Liczba scenariuszy (domyślnie 100).
        noise_pct: Procent szumu (domyślnie 1.0% = tolerancja E24).

    Returns:
        Słownik:
          - recalls: Lista recalii z n_scenarios
          - mean_recall, std_recall
          - confidence_table: [(percentyl, min_recall)]
          - hw_ready: bool — czy system przechodzi

    Przykład:
        >>> results = hil_simulation(model, test_loader)
        >>> print(f"P(recall ≥ 85%) = {results['prob_recall_ge_85']:.1%}")
    """
    model.eval()
    model.enable_mismatch(False)
    model.set_quantize_mode("none")

    weight_params = [
        model.w_n1, model.w_n2, model.w_n3_from_n1,
        model.w_n3_from_n2, model.w_inh, model.w_inh_to_n3,
    ]
    original_weights = [p.data.clone() for p in weight_params]

    recalls = []

    print(f"\n[HIL] Symulacja {n_scenarios} scenariuszy z ±{noise_pct}% szumem rezystorów...")

    for scenario in tqdm(range(n_scenarios), desc="HIL simulation"):
        # Dodaj szum ±noise_pct% do każdej wagi
        for param, orig in zip(weight_params, original_weights):
            noise = torch.randn_like(orig) * (noise_pct / 100.0)
            param.data.copy_(orig * (1.0 + noise))

        # Ewaluacja
        all_preds = []
        all_targets = []
        for spikes, labels in test_loader:
            spikes = spikes.to(DEVICE)
            trigger, _ = model(spikes)
            all_preds.append(trigger.cpu())
            all_targets.append(labels)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        r = recall_score(all_preds, all_targets)
        recalls.append(r)

    # Przywróć oryginalne wagi
    for param, orig in zip(weight_params, original_weights):
        param.data.copy_(orig)

    recalls_np = np.array(recalls)

    # Statystyki
    confidence_table = []
    for threshold in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        prob = float(np.mean(recalls_np >= threshold))
        confidence_table.append((threshold, prob))

    result = {
        "recalls": recalls,
        "mean_recall": float(np.mean(recalls_np)),
        "std_recall": float(np.std(recalls_np)),
        "min_recall": float(np.min(recalls_np)),
        "max_recall": float(np.max(recalls_np)),
        "percentile_5": float(np.percentile(recalls_np, 5)),
        "percentile_25": float(np.percentile(recalls_np, 25)),
        "median_recall": float(np.median(recalls_np)),
        "confidence_table": confidence_table,
        "prob_recall_ge_85": float(np.mean(recalls_np >= 0.85)),
        "prob_recall_ge_80": float(np.mean(recalls_np >= 0.80)),
    }

    # Raport
    print(f"\n{'=' * 60}")
    print(f"  HARDWARE-IN-THE-LOOP VALIDATION REPORT")
    print(f"  {n_scenarios} scenariuszy, ±{noise_pct}% szum rezystorów")
    print(f"{'=' * 60}")
    print(f"  Mean recall:     {result['mean_recall']:.4f} ± {result['std_recall']:.4f}")
    print(f"  Median recall:   {result['median_recall']:.4f}")
    print(f"  Min / Max:       [{result['min_recall']:.4f}, {result['max_recall']:.4f}]")
    print(f"  5th percentile:  {result['percentile_5']:.4f}")
    print()
    print(f"  {'Próg recall':<15} {'Prawdop. osiągnięcia':>22}")
    print(f"  {'-' * 37}")
    for threshold, prob in confidence_table:
        emoji = "✓" if prob >= 0.90 else "⚠" if prob >= 0.50 else "✗"
        print(f"  {emoji} recall ≥ {threshold:.0%}     →  {prob:.1%}")
    print(f"{'=' * 60}")

    return result


def identify_critical_synapses(
    sensitivities: Dict[str, float],
    threshold_pct: float = 5.0,
) -> List[Dict[str, Any]]:
    """Identyfikuje synapsy wymagające potencjometrów cyfrowych MCP4151.

    Kryterium: wrażliwość > threshold_pct% recall na ±1% wagi.

    Args:
        sensitivities: Wynik sensitivity_analysis().
        threshold_pct: Próg wrażliwości (domyślnie 5%).

    Returns:
        Lista słowników z informacjami o krytycznych synapsach.

    Przykład:
        >>> critical = identify_critical_synapses(sens)
        >>> for s in critical:
        ...     print(f"{s['name']}: wrażliwość {s['sensitivity']:.1f}%")
    """
    critical = []
    for name, deg in sensitivities.items():
        if deg > threshold_pct:
            critical.append({
                "name": name,
                "sensitivity": deg,
                "recommendation": "MCP4151 potencjometr cyfrowy",
            })

    if critical:
        print(f"\n[KRYTYCZNE] {len(critical)} synaps wymaga MCP4151:")
        for s in critical:
            print(f"  ⚠ {s['name']}: {s['sensitivity']:.1f}% wrażliwości recall")
    else:
        print("[INFO] Żadna synapsa nie wymaga MCP4151 — wszystkie stabilne z rezystorami E24.")

    return critical


def generate_mcp4151_table(
    model: GlassBreakSNN,
    critical_synapses: Optional[List[str]] = None,
    mcp4151_r_total: float = 100_000.0,
    save_path: Optional[str] = None,
) -> str:
    """Generuje tabelę komend SPI dla potencjometrów cyfrowych MCP4151.

    MCP4151 ma 256 kroków (wiper value 0-255).
    R_wiper = R_total × (wiper_value / 256) + R_wiper_typ (75Ω).
    Bajty SPI: [address_byte, data_byte] — write to wiper: 0x00 + data.

    Args:
        model: Model SNN.
        critical_synapses: Lista nazw synaps do MCP4151 (None = wszystkie).
        mcp4151_r_total: Rezystancja końcowa MCP4151 (domyślnie 100kΩ).
        save_path: Ścieżka do zapisu CSV.

    Returns:
        Ścieżka do pliku CSV.

    Przykład:
        >>> path = generate_mcp4151_table(model, ["w_n3_from_n1", "w_n3_from_n2"])
    """
    if save_path is None:
        save_path = str(PATH_CONFIG.output_dir / "mcp4151_table.csv")

    weights = model.get_weights_dict()

    if critical_synapses is None:
        # Wszystkie wagi (na wszelki wypadek)
        critical_synapses = [k for k in weights.keys() if k != "w_inh_to_n3"]

    R_W = 75.0  # Typowa rezystancja wipera MCP4151

    rows = [
        ["Neuron", "Synapsa", "Waga", "R_target_ohm", "Wiper_value_0_255",
         "R_actual_ohm", "SPI_byte0_hex", "SPI_byte1_hex", "SPI_command"],
    ]

    for syn_name in critical_synapses:
        if syn_name not in weights:
            continue

        w_val = abs(weights[syn_name])
        r_target = weight_to_resistance(w_val)

        # Clamp do zakresu MCP4151
        r_target_clamped = max(R_W, min(r_target, mcp4151_r_total + R_W))

        # Oblicz wiper value: R = R_W + R_total × (wiper / 256)
        # wiper = (R - R_W) / R_total × 256
        wiper_value = int(round((r_target_clamped - R_W) / mcp4151_r_total * 256))
        wiper_value = max(0, min(255, wiper_value))

        # Rzeczywista rezystancja po ustawieniu wipera
        r_actual = R_W + mcp4151_r_total * (wiper_value / 256.0)

        # SPI command: byte0 = 0x00 (write to wiper 0), byte1 = wiper value
        spi_byte0 = 0x00
        spi_byte1 = wiper_value

        # Nazwa neuronu
        neuron = syn_name.replace("w_", "").split("_")[0].upper()
        if "n3_from" in syn_name:
            neuron = "N3"
        elif "inh" in syn_name:
            neuron = "N_INH"

        rows.append([
            neuron,
            syn_name,
            f"{w_val:.4f}",
            f"{r_target:.0f}",
            f"{wiper_value}",
            f"{r_actual:.0f}",
            f"0x{spi_byte0:02X}",
            f"0x{spi_byte1:02X}",
            f"SPI.transfer(0x{spi_byte0:02X}); SPI.transfer(0x{spi_byte1:02X});",
        ])

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"\n[MCP4151] Tabela programowania zapisana do {save_path}")
    print(f"[MCP4151] {len(rows) - 1} potencjometrów do zaprogramowania.")

    # Pretty print
    print(f"\n  {'Neuron':<8} {'Synapsa':<18} {'R_target':>10} {'Wiper':>6} {'R_actual':>10} {'SPI Command'}")
    print(f"  {'-' * 72}")
    for row in rows[1:]:
        print(f"  {row[0]:<8} {row[1]:<18} {row[3]:>10}Ω {row[4]:>6} {row[5]:>10}Ω {row[8]}")

    return save_path
