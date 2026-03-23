# -*- coding: utf-8 -*-
"""
Globalna konfiguracja projektu SNN — stałe, parametry hardware, hiperparametry treningu.

Wszystkie wartości są zdefiniowane centralnie żeby uniknąć magic numbers w kodzie.
Zmiany parametrów hardware (np. zakres rezystorów, seria E24) powinny być robione TUTAJ.
"""

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

# =============================================================================
# SEED — reprodukowalność wyników
# =============================================================================
RANDOM_SEED: int = 42

def set_global_seed(seed: int = RANDOM_SEED) -> None:
    """Ustawia seed dla wszystkich generatorów losowych.

    Args:
        seed: Wartość ziarna losowości.

    Przykład:
        >>> set_global_seed(42)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# SERIA E24 — wartości rezystorów (24 wartości na dekadę)
# =============================================================================
# Bazowe wartości E24 (mnożniki w jednej dekadzie, np. 1.0 = 10Ω, 100Ω, 1kΩ...)
E24_BASE: List[float] = [
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0,
    2.2, 2.4, 2.7, 3.0, 3.3, 3.6, 3.9, 4.3,
    4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1
]

def generate_e24_full_range(r_min: float = 10.0, r_max: float = 10_000_000.0) -> np.ndarray:
    """Generuje pełną listę wartości E24 w podanym zakresie oporności.

    Seria E24 ma 24 wartości na dekadę. Funkcja generuje wartości od r_min do r_max
    uwzględniając wszystkie dekady (10Ω, 100Ω, 1kΩ, ..., 10MΩ).

    Args:
        r_min: Minimalna rezystancja w Ohmach.
        r_max: Maksymalna rezystancja w Ohmach.

    Returns:
        Posortowana tablica numpy z wartościami E24 w zakresie [r_min, r_max].

    Przykład:
        >>> vals = generate_e24_full_range(10e3, 470e3)
        >>> print(f"Znaleziono {len(vals)} wartości E24")
    """
    values = []
    # Dekady: 10^1, 10^2, ..., 10^7
    for decade_exp in range(1, 8):
        multiplier = 10 ** decade_exp
        for base in E24_BASE:
            r = base * multiplier
            if r_min <= r <= r_max:
                values.append(r)
    return np.sort(np.array(values, dtype=np.float64))

# Pełna lista E24 w zakresie synaps PCB (10kΩ – 470kΩ)
E24_SYNAPSE_RANGE: np.ndarray = generate_e24_full_range(10_000.0, 470_000.0)

# Pełna lista E24 znormalizowana do [0, 1] (do użytku w treningu)
E24_FULL: np.ndarray = generate_e24_full_range(10.0, 10_000_000.0)
E24_NORMALIZED: np.ndarray = np.sort(np.unique(
    np.array([b / 10.0 for b in E24_BASE], dtype=np.float64)
))  # Wartości 0.10 ... 0.91 — bazowe mnożniki znormalizowane

# Tensor PyTorch do użytku w forward pass (zaokrąglanie)
E24_NORMALIZED_TENSOR: torch.Tensor = torch.tensor(
    E24_NORMALIZED, dtype=torch.float32
)


# =============================================================================
# PARAMETRY HARDWARE — ograniczenia PCB
# =============================================================================
@dataclass
class HardwareConfig:
    """Parametry ograniczeń sprzętowych analogowego PCB.

    Attributes:
        r_in: Rezystancja wejściowa neuronu (Ohm).
        r_syn_min: Minimalna rezystancja synaptyczna (Ohm).
        r_syn_max: Maksymalna rezystancja synaptyczna (Ohm).
        v_th_step: Krok progu V_th potencjometru (Volt).
        v_th_min: Minimalny próg V_th (Volt).
        v_th_max: Maksymalny próg V_th (Volt).
        vcc: Napięcie zasilania (Volt).
        resistor_tolerance: Tolerancja rezystorów (ułamek, np. 0.01 = 1%).
        thermal_drift_mv: Dryf termiczny progu (miliVolt).
        max_synapse_current_ua: Maksymalny prąd synapsy (µA).
        mcp4151_steps: Liczba kroków potencjometru cyfrowego (0-255).
    """
    r_in: float = 100_000.0          # 100kΩ
    r_syn_min: float = 10_000.0      # 10kΩ
    r_syn_max: float = 470_000.0     # 470kΩ
    v_th_step: float = 0.05          # 50mV krok
    v_th_min: float = 0.3            # V
    v_th_max: float = 1.5            # V
    vcc: float = 5.0                 # V
    resistor_tolerance: float = 0.01  # ±1%
    thermal_drift_mv: float = 5.0    # ±5mV
    max_synapse_current_ua: float = 100.0
    mcp4151_steps: int = 256         # 0-255


# =============================================================================
# PARAMETRY NEURONU LIF
# =============================================================================
@dataclass
class LIFConfig:
    """Parametry neuronu Leaky Integrate-and-Fire.

    Attributes:
        tau_m: Stała czasowa membrany (sekundy).
        v_th_default: Domyślny próg wyzwalania (znormalizowany 0-1).
        v_rest: Napięcie spoczynkowe (znormalizowany).
        dt: Krok czasowy symulacji (sekundy).
        refractory_ms: Okres refrakcji (milisekundy).
        beta: Współczynnik decay membrany = exp(-dt/tau_m).
    """
    tau_m: float = 0.020             # 20ms — kompromis między 10-100ms
    v_th_default: float = 0.65       # ~1.2V w normalizacji przy VCC=5V → 0.24, ale spec mówi 0.65
    v_rest: float = 0.0
    dt: float = 0.001               # 1ms krok
    refractory_ms: float = 2.0

    @property
    def beta(self) -> float:
        """Współczynnik decay membrany (beta = exp(-dt/tau_m))."""
        return float(np.exp(-self.dt / self.tau_m))


# =============================================================================
# PARAMETRY POSZCZEGÓLNYCH NEURONÓW (baseline z symulacji)
# =============================================================================
@dataclass
class NeuronParams:
    """Parametry jednego neuronu z baseline symulacji.

    Attributes:
        name: Nazwa neuronu (N1, N2, N3, N_inh).
        w_in: Waga(-i) wejściowa (lista, bo N3 ma 2 wejścia).
        v_th: Próg wyzwalania (znormalizowany 0-1).
        description: Opis roli neuronu.
        qat_bits: Precyzja kwantyzacji (bity) dla QAT.
    """
    name: str
    w_in: List[float]
    v_th: float
    description: str
    qat_bits: int = 5

# Baseline parametry z symulacji (branch sim_train)
BASELINE_NEURONS = {
    "N1": NeuronParams(
        name="N1",
        w_in=[0.7],
        v_th=0.6,
        description="Detektor wysokiej częstotliwości (>2kHz)",
        qat_bits=5,
    ),
    "N2": NeuronParams(
        name="N2",
        w_in=[0.5],
        v_th=0.5,
        description="Detektor temporal (pattern burst)",
        qat_bits=5,
    ),
    "N3": NeuronParams(
        name="N3",
        w_in=[0.72, 0.44],  # wejścia z N1 i N2
        v_th=0.8,
        description="Neuron wyjściowy — zbiera N1+N2 → trigger",
        qat_bits=6,
    ),
    "N_inh": NeuronParams(
        name="N_inh",
        w_in=[0.33],
        v_th=0.4,
        description="Neuron hamujący — reaguje na HVAC/szumy <500Hz",
        qat_bits=4,
    ),
}


# =============================================================================
# PARAMETRY AUDIO / SPIKE ENCODING
# =============================================================================
@dataclass
class AudioConfig:
    """Parametry przetwarzania audio i enkodowania spike'ów.

    Attributes:
        sample_rate: Częstotliwość próbkowania (Hz).
        bandpass_low: Dolna częstotliwość filtru pasmowego (Hz).
        bandpass_high: Górna częstotliwość filtru pasmowego (Hz).
        rms_window_ms: Szerokość okna RMS (ms).
        rms_hop_ms: Krok okna RMS (ms).
        n_mfcc: Liczba współczynników MFCC.
        rate_min_hz: Minimalna częstotliwość rate coding (Hz).
        rate_max_hz: Maksymalna częstotliwość rate coding (Hz).
        ttfs_window_ms: Okno TTFS (ms).
        spike_duration_s: Czas trwania pojedynczego spike'a (sekundy).
    """
    sample_rate: int = 22050
    bandpass_low: float = 500.0
    bandpass_high: float = 8000.0
    rms_window_ms: float = 10.0
    rms_hop_ms: float = 5.0
    n_mfcc: int = 13
    rate_min_hz: float = 5.0
    rate_max_hz: float = 200.0
    ttfs_window_ms: float = 20.0
    spike_duration_s: float = 0.001  # 1ms


# =============================================================================
# HIPERPARAMETRY TRENINGU
# =============================================================================
@dataclass
class TrainConfig:
    """Hiperparametry treningu HAT i QAT.

    Attributes:
        hat_epochs: Liczba epok HAT.
        qat_epochs: Liczba epok QAT fine-tuning.
        batch_size: Rozmiar batcha.
        learning_rate: Tempo uczenia (Adam).
        lambda_recall: Waga kary za niski recall.
        lambda_hw: Waga regularyzacji E24.
        recall_target: Docelowy recall (ułamek).
        temp_start: Temperatura początku annealing (Gumbel-softmax).
        temp_end: Temperatura końca annealing.
        mismatch_weight_pct: Szum mismatch wag (procent).
        mismatch_vth_mv: Szum mismatch progu (mV).
        thermal_drift_pct: Dryf termiczny do ewaluacji (procent).
        thermal_drift_runs: Liczba powtórzeń symulacji dryfu.
        v_th_sweep_min: Min wartość sweep progu.
        v_th_sweep_max: Max wartość sweep progu.
        v_th_sweep_step: Krok sweep progu.
        calibration_samples: Liczba próbek kalibracyjnych (QAT).
    """
    hat_epochs: int = 50
    qat_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 1e-3
    lambda_recall: float = 2.0
    lambda_hw: float = 0.5
    recall_target: float = 0.85
    temp_start: float = 5.0
    temp_end: float = 0.1
    mismatch_weight_pct: float = 1.0    # ±1%
    mismatch_vth_mv: float = 5.0        # ±5mV
    thermal_drift_pct: float = 2.0      # ±2% dla ewaluacji
    thermal_drift_runs: int = 100
    v_th_sweep_min: float = 0.3
    v_th_sweep_max: float = 1.5
    v_th_sweep_step: float = 0.05
    calibration_samples: int = 50


# =============================================================================
# ŚCIEŻKI PROJEKTU
# =============================================================================
@dataclass
class PathConfig:
    """Ścieżki do plików i katalogów projektu.

    Attributes:
        project_root: Katalog główny projektu.
        data_dir: Katalog z danymi (ESC-50).
        output_dir: Katalog wyjściowy (wagi, wykresy, raporty).
        checkpoint_dir: Katalog z checkpointami modelu.
    """
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    output_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "output")
    checkpoint_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "checkpoints")

    def __post_init__(self) -> None:
        """Tworzy katalogi jeśli nie istnieją."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# ESC-50 — mapowanie klas
# =============================================================================
# ESC-50 ma 50 klas, każda po 40 próbek = 2000 total
# Klasa "glass_breaking" to ID 38 (fold 1-5)
ESC50_GLASS_BREAKING_CLASS: int = 38
ESC50_NUM_CLASSES: int = 50
ESC50_SAMPLES_PER_CLASS: int = 40
ESC50_URL: str = "https://github.com/karoldvl/ESC-50/archive/master.zip"

# Klasy tła / negatywne do analizy confusion matrix
ESC50_HVAC_CLASSES: List[int] = [
    36,  # air_conditioner
    37,  # car_horn (blisko, ale inny charakter)
    41,  # siren
]

ESC50_BACKGROUND_CLASSES: List[int] = [
    40,  # hand_saw
    43,  # engine
    45,  # train
]


# =============================================================================
# DOMYŚLNE INSTANCJE KONFIGURACJI
# =============================================================================
HW_CONFIG = HardwareConfig()
LIF_CONFIG = LIFConfig()
AUDIO_CONFIG = AudioConfig()
TRAIN_CONFIG = TrainConfig()
PATH_CONFIG = PathConfig()

# Urządzenie (CPU — studenci nie mają GPU)
DEVICE = torch.device("cpu")
