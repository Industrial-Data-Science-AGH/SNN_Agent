# -*- coding: utf-8 -*-
"""
Eksport wag do formatu PCB — JSON, CSV, Arduino .h header.

Konwertuje wytrenowane wagi na:
1. JSON z wagami, rezystancjami E24, progami V_th
2. CSV z tabelą wag
3. Arduino .h z #define parametrami enkodera
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from snn_pipeline.config import (
    AUDIO_CONFIG,
    HW_CONFIG,
    PATH_CONFIG,
)
from snn_pipeline.e24_quantizer import (
    quantize_to_e24,
    weight_to_nearest_e24_resistance,
    weight_to_resistance,
)
from snn_pipeline.snn_model import GlassBreakSNN


def export_weights_json(
    model: GlassBreakSNN,
    metrics: Dict[str, float],
    version: str = "hat_qat_v1",
    save_path: Optional[str] = None,
) -> str:
    """Eksportuje wagi modelu do formatu JSON gotowego na PCB.

    Format:
    ```json
    {
      "version": "hat_qat_v1",
      "neurons": {
        "N1": {"w_in": 0.68, "V_th": 0.62, "R_syn_ohm": 47058, "R_syn_E24": 47000},
        ...
      },
      "metrics": {"precision": 1.0, "recall": 0.87, "f1": 0.93}
    }
    ```

    Args:
        model: Wytrenowany model SNN.
        metrics: Słownik metryk (precision, recall, f1, ...).
        version: Etykieta wersji.
        save_path: Ścieżka do zapisu. Domyślnie: output/weights.json.

    Returns:
        Ścieżka do zapisanego pliku.

    Przykład:
        >>> path = export_weights_json(model, {"precision": 1.0, "recall": 0.87})
    """
    if save_path is None:
        save_path = str(PATH_CONFIG.output_dir / "weights.json")

    weights = model.get_weights_dict()
    thresholds = model.get_thresholds_dict()

    # N1
    r_n1_exact, r_n1_e24 = weight_to_nearest_e24_resistance(weights["w_n1"])
    # N2
    r_n2_exact, r_n2_e24 = weight_to_nearest_e24_resistance(weights["w_n2"])
    # N3 ma dwa wejścia
    r_n3_1_exact, r_n3_1_e24 = weight_to_nearest_e24_resistance(weights["w_n3_from_n1"])
    r_n3_2_exact, r_n3_2_e24 = weight_to_nearest_e24_resistance(weights["w_n3_from_n2"])
    # N_inh
    r_inh_exact, r_inh_e24 = weight_to_nearest_e24_resistance(abs(weights["w_inh"]))

    data = {
        "version": version,
        "hardware": {
            "R_in_ohm": HW_CONFIG.r_in,
            "VCC_V": HW_CONFIG.vcc,
            "resistor_tolerance_pct": HW_CONFIG.resistor_tolerance * 100,
        },
        "neurons": {
            "N1": {
                "w_in": round(weights["w_n1"], 4),
                "V_th": round(thresholds["vth_n1"], 4),
                "V_th_analog_V": round(thresholds["vth_n1"] * HW_CONFIG.vcc, 3),
                "R_syn_ohm": round(r_n1_exact),
                "R_syn_E24": round(r_n1_e24),
                "description": "Detektor wysokiej częstotliwości (>2kHz)",
            },
            "N2": {
                "w_in": round(weights["w_n2"], 4),
                "V_th": round(thresholds["vth_n2"], 4),
                "V_th_analog_V": round(thresholds["vth_n2"] * HW_CONFIG.vcc, 3),
                "R_syn_ohm": round(r_n2_exact),
                "R_syn_E24": round(r_n2_e24),
                "description": "Detektor temporal (burst pattern)",
            },
            "N3": {
                "w_in_N1": round(weights["w_n3_from_n1"], 4),
                "w_in_N2": round(weights["w_n3_from_n2"], 4),
                "V_th": round(thresholds["vth_n3"], 4),
                "V_th_analog_V": round(thresholds["vth_n3"] * HW_CONFIG.vcc, 3),
                "R_syn_from_N1_ohm": round(r_n3_1_exact),
                "R_syn_from_N1_E24": round(r_n3_1_e24),
                "R_syn_from_N2_ohm": round(r_n3_2_exact),
                "R_syn_from_N2_E24": round(r_n3_2_e24),
                "description": "Neuron wyjściowy — trigger",
            },
            "N_inh": {
                "w_in": round(abs(weights["w_inh"]), 4),
                "w_inhibition_to_N3": round(weights["w_inh_to_n3"], 4),
                "V_th": round(thresholds["vth_inh"], 4),
                "V_th_analog_V": round(thresholds["vth_inh"] * HW_CONFIG.vcc, 3),
                "R_inh_ohm": round(r_inh_exact),
                "R_inh_E24": round(r_inh_e24),
                "description": "Neuron hamujący — HVAC/szumy <500Hz",
            },
        },
        "metrics": {
            "precision": round(metrics.get("precision", 0.0), 4),
            "recall": round(metrics.get("recall", 0.0), 4),
            "f1": round(metrics.get("f1", 0.0), 4),
            "accuracy": round(metrics.get("accuracy", 0.0), 4),
        },
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[Export] Wagi JSON zapisane do {save_path}")
    return save_path


def export_weights_csv(
    model: GlassBreakSNN,
    save_path: Optional[str] = None,
) -> str:
    """Eksportuje wagi do CSV.

    Kolumny: Neuron, Synapsa, Waga, R_syn_exact, R_syn_E24, Błąd_pct, V_th

    Args:
        model: Model SNN.
        save_path: Ścieżka do zapisu.

    Returns:
        Ścieżka do zapisanego pliku.

    Przykład:
        >>> path = export_weights_csv(model)
    """
    if save_path is None:
        save_path = str(PATH_CONFIG.output_dir / "weights.csv")

    weights = model.get_weights_dict()
    thresholds = model.get_thresholds_dict()

    rows = [
        ["Neuron", "Synapsa", "Waga", "R_syn_exact_ohm", "R_syn_E24_ohm", "Blad_pct", "V_th"],
    ]

    synapse_map = {
        "w_n1": ("N1", "Input→N1", "vth_n1"),
        "w_n2": ("N2", "Input→N2", "vth_n2"),
        "w_n3_from_n1": ("N3", "N1→N3", "vth_n3"),
        "w_n3_from_n2": ("N3", "N2→N3", "vth_n3"),
        "w_inh": ("N_inh", "Input→N_inh", "vth_inh"),
        "w_inh_to_n3": ("N_inh→N3", "N_inh→N3", "vth_n3"),
    }

    for w_name, (neuron, synapse, vth_key) in synapse_map.items():
        w_val = weights[w_name]
        vth_val = thresholds.get(vth_key, 0.0)
        r_exact, r_e24 = weight_to_nearest_e24_resistance(abs(w_val))
        err_pct = abs(r_exact - r_e24) / max(r_exact, 1) * 100

        rows.append([
            neuron, synapse, f"{w_val:.4f}",
            f"{r_exact:.0f}", f"{r_e24:.0f}", f"{err_pct:.2f}",
            f"{vth_val:.4f}",
        ])

    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"[Export] Wagi CSV zapisane do {save_path}")
    return save_path


def generate_arduino_header(
    model: GlassBreakSNN,
    save_path: Optional[str] = None,
) -> str:
    """Generuje plik .h dla Arduino z parametrami enkodera.

    Zawiera #define dla:
    - RC_NOISE_FLOOR — próg szumu rate coding
    - RC_MAX_RATE_HZ — max częstotliwość
    - TTFS_THRESHOLD — próg TTFS
    - TTFS_WINDOW_MS — okno TTFS
    - Wagi i progi V_th jako stałe

    Args:
        model: Model SNN.
        save_path: Ścieżka do zapisu.

    Returns:
        Ścieżka do zapisanego pliku.

    Przykład:
        >>> path = generate_arduino_header(model)
    """
    if save_path is None:
        save_path = str(PATH_CONFIG.output_dir / "snn_config.h")

    weights = model.get_weights_dict()
    thresholds = model.get_thresholds_dict()

    # Oblicz wartości enkodera
    noise_floor = int(AUDIO_CONFIG.rate_min_hz * 255 / AUDIO_CONFIG.rate_max_hz)
    max_rate = int(AUDIO_CONFIG.rate_max_hz)
    ttfs_threshold = int(0.05 * 1024)  # ADC 10-bit próg szumu
    ttfs_window = int(AUDIO_CONFIG.ttfs_window_ms)

    header_content = f"""/*
 * snn_config.h — Automatycznie wygenerowany plik konfiguracyjny SNN
 *
 * Wagi i progi wytrenowanej sieci SNN (HAT+QAT).
 * NIE EDYTUJ RĘCZNIE — plik generowany przez snn_pipeline/export.py.
 *
 * Wersja: hat_qat_v1
 */

#ifndef SNN_CONFIG_H
#define SNN_CONFIG_H

// =============================================================================
// PARAMETRY ENKODERA (Rate Coding / TTFS)
// =============================================================================
#define RC_NOISE_FLOOR    {noise_floor}     // Próg szumu ADC (0-255)
#define RC_MAX_RATE_HZ    {max_rate}     // Max częstotliwość spike'ów (Hz)
#define TTFS_THRESHOLD    {ttfs_threshold}      // Próg wyzwalania TTFS (ADC units)
#define TTFS_WINDOW_MS    {ttfs_window}      // Okno TTFS (ms)
#define SAMPLE_PERIOD_MS  10       // Okres próbkowania SPI (ms)

// =============================================================================
// WAGI SYNAPTYCZNE (znormalizowane 0.0–1.0, × 100 dla integer arytmetyki)
// =============================================================================
#define W_N1_INPUT        {int(abs(weights['w_n1']) * 100)}      // Waga Input→N1 (×100)
#define W_N2_INPUT        {int(abs(weights['w_n2']) * 100)}      // Waga Input→N2 (×100)
#define W_N3_FROM_N1      {int(abs(weights['w_n3_from_n1']) * 100)}      // Waga N1→N3 (×100)
#define W_N3_FROM_N2      {int(abs(weights['w_n3_from_n2']) * 100)}      // Waga N2→N3 (×100)
#define W_INH_INPUT       {int(abs(weights['w_inh']) * 100)}      // Waga Input→N_inh (×100)
#define W_INH_TO_N3       {int(abs(weights['w_inh_to_n3']) * 100)}      // Waga hamowania N_inh→N3 (×100, UJEMNA)

// =============================================================================
// PROGI WYZWALANIA V_th (znormalizowane × 100)
// =============================================================================
#define VTH_N1            {int(thresholds['vth_n1'] * 100)}      // Próg N1 (×100)
#define VTH_N2            {int(thresholds['vth_n2'] * 100)}      // Próg N2 (×100)
#define VTH_N3            {int(thresholds['vth_n3'] * 100)}      // Próg N3 (×100)
#define VTH_INH           {int(thresholds['vth_inh'] * 100)}      // Próg N_inh (×100)

// =============================================================================
// PROGI ANALOGOWE (mV, przy VCC=5V)
// =============================================================================
#define VTH_N1_MV         {int(thresholds['vth_n1'] * HW_CONFIG.vcc * 1000)}     // Próg N1 (mV)
#define VTH_N2_MV         {int(thresholds['vth_n2'] * HW_CONFIG.vcc * 1000)}     // Próg N2 (mV)
#define VTH_N3_MV         {int(thresholds['vth_n3'] * HW_CONFIG.vcc * 1000)}     // Próg N3 (mV)
#define VTH_INH_MV        {int(thresholds['vth_inh'] * HW_CONFIG.vcc * 1000)}     // Próg N_inh (mV)

// =============================================================================
// SPI PROTOCOL
// =============================================================================
#define SPI_MAGIC_BITS    0b101    // Magic number (3 bity) — weryfikacja pakietu
#define SPI_PACKET_SIZE   2        // Rozmiar pakietu (bajty)

#endif // SNN_CONFIG_H
"""

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(header_content)

    print(f"[Export] Arduino header zapisany do {save_path}")
    return save_path
