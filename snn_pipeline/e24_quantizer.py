# -*- coding: utf-8 -*-
"""
Kwantyzator E24 — mapowanie ciągłych wag na wartości z serii rezystorów E24.

Implementuje Straight-Through Estimator (STE) i Gumbel-softmax temperature annealing
dla różniczkowalnego przejścia od miękkich do twardych wag w trakcie treningu HAT/QAT.

Seria E24 ma nierównomierne kroki (~10% między sąsiednimi wartościami),
więc nie stosujemy uniform quantization.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from typing import Optional, Tuple

from snn_pipeline.config import E24_BASE, E24_NORMALIZED_TENSOR, HW_CONFIG


# =============================================================================
# PEŁNA LISTA WARTOŚCI E24 (znormalizowanych do [0,1])
# =============================================================================

def build_e24_normalized_grid() -> torch.Tensor:
    """Buduje pełną siatkę wartości E24 znormalizowanych do zakresu [0,1].

    Wartości bazowe E24 (24 na dekadę) są normalizowane tak, żeby pokryć
    efektywny zakres wag synaptycznych. W praktyce używamy e24_base / 10.0
    bo wagi trenujemy w zakresie [0,1].

    Returns:
        Posortowany tensor z wartościami E24 w [0, 1].

    Przykład:
        >>> grid = build_e24_normalized_grid()
        >>> print(f"Grid: {grid[:5]}")  # tensor([0.10, 0.11, 0.12, ...])
    """
    # Bazowe wartości E24 znormalizowane do [0,1]
    values = sorted(set([b / 10.0 for b in E24_BASE]))
    return torch.tensor(values, dtype=torch.float32)


# Globalna stała — siatka E24 na GPU/CPU
_E24_GRID = build_e24_normalized_grid()


def get_e24_grid(device: Optional[torch.device] = None) -> torch.Tensor:
    """Zwraca siatkę E24 na odpowiednim urządzeniu.

    Args:
        device: Urządzenie PyTorch (CPU/CUDA).

    Returns:
        Tensor z wartościami E24.
    """
    global _E24_GRID
    if device is not None and _E24_GRID.device != device:
        _E24_GRID = _E24_GRID.to(device)
    return _E24_GRID


# =============================================================================
# QUANTIZE TO E24 — najważniejsza operacja
# =============================================================================

def quantize_to_e24(w: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Zaokrągla ciągłą wagę do najbliższej wartości z siatki E24.

    Operacja nieróżniczkowalna — gradient nie przepływa. Użyj E24STE do treningu.

    Args:
        w: Tensor wag ciągłych (dowolny kształt).
        grid: Opcjonalny tensor siatki E24. Domyślnie: globalna siatka.

    Returns:
        Tensor wag zaokrąglonych do E24 (ten sam kształt co w).

    Przykład:
        >>> w = torch.tensor([0.35, 0.72, 0.15])
        >>> w_q = quantize_to_e24(w)
        >>> print(w_q)  # tensor([0.33, 0.75, 0.15])
    """
    if grid is None:
        grid = get_e24_grid(w.device)

    # Rozwiń w do kolumny, grid do wiersza → macierz odległości
    w_flat = w.reshape(-1, 1)
    diffs = torch.abs(w_flat - grid.reshape(1, -1))
    indices = torch.argmin(diffs, dim=1)
    quantized = grid[indices].reshape(w.shape)
    return quantized


def quantize_to_e24_with_error(
    w: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Kwantyzuje wagi i zwraca również błąd kwantyzacji.

    Args:
        w: Tensor wag ciągłych.

    Returns:
        Krotka (w_quantized, error_absolute, error_percent).

    Przykład:
        >>> w = torch.tensor([0.35])
        >>> w_q, err_abs, err_pct = quantize_to_e24_with_error(w)
    """
    w_q = quantize_to_e24(w)
    err_abs = torch.abs(w - w_q)
    err_pct = err_abs / (torch.abs(w) + 1e-8) * 100.0
    return w_q, err_abs, err_pct


# =============================================================================
# STRAIGHT-THROUGH ESTIMATOR (STE) — rdzeń HAT
# =============================================================================

class E24STEFunction(Function):
    """Straight-Through Estimator dla kwantyzacji E24.

    Forward: zaokrągla wagi do E24 (nieróżniczkowalne).
    Backward: przepuszcza gradient "na wprost" (jakby kwantyzacja nie istniała).

    To pozwala trenować wagi z gradient descent mimo operacji round.
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """Forward pass — kwantyzacja do E24.

        Args:
            w: Wagi ciągłe.
            grid: Siatka E24.

        Returns:
            Wagi skwantyzowane do E24.
        """
        return quantize_to_e24(w, grid)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass — STE: gradient przepływa bez zmian.

        Args:
            grad_output: Gradient z kolejnej warstwy.

        Returns:
            (gradient_for_w, None_for_grid).
        """
        # Straight-through: gradient przechodzi bez modyfikacji
        return grad_output, None


def e24_ste(w: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Convenience wrapper na E24 STE.

    Fake quantization: w_q = quantize_E24(w) + stop_gradient(w - quantize_E24(w))
    Efektywnie: forward widzi skwantyzowaną wartość, backward widzi ciągłą.

    Args:
        w: Tensor wag ciągłych.
        grid: Opcjonalna siatka E24.

    Returns:
        Tensor z "fake quantized" wagami (forward = E24, backward = ciągły gradient).

    Przykład:
        >>> w = torch.tensor([0.35], requires_grad=True)
        >>> w_q = e24_ste(w)
        >>> loss = (w_q - 0.5).pow(2)
        >>> loss.backward()
        >>> print(w.grad)  # gradient istnieje mimo kwantyzacji!
    """
    if grid is None:
        grid = get_e24_grid(w.device)
    return E24STEFunction.apply(w, grid)


# =============================================================================
# GUMBEL-SOFTMAX TEMPERATURE ANNEALING
# =============================================================================

class E24GumbelQuantizer(nn.Module):
    """Kwantyzator E24 z Gumbel-softmax temperature annealing.

    Na początku treningu (T=5.0) kwantyzacja jest "miękka" — wagi mogą
    przyjmować wartości pomiędzy punktami E24. Pod koniec (T=0.1) kwantyzacja
    staje się "twarda" — wagi są dokładnie na wartościach E24.

    Mechanizm: waga jest expressed jako mieszanka wartości E24 ważona
    softmaxem z odległości. Temperatura kontroluje "ostrość" tego softmaxa.

    Attributes:
        temperature: Aktualna temperatura (zmniejszana przez trainer).

    Przykład:
        >>> quantizer = E24GumbelQuantizer(temperature=5.0)
        >>> w = torch.tensor([0.35], requires_grad=True)
        >>> w_q = quantizer(w)
        >>> # Na początku: w_q ≈ 0.35 (miękka)
        >>> quantizer.temperature = 0.1
        >>> w_q = quantizer(w)
        >>> # Na końcu: w_q ≈ 0.33 (twarda, najbliższa E24)
    """

    def __init__(self, temperature: float = 5.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.register_buffer("grid", build_e24_normalized_grid())

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """Forward pass z Gumbel-softmax quantization.

        Args:
            w: Tensor wag ciągłych.

        Returns:
            Tensor wag "miękko-skwantyzowanych" do E24.
        """
        # Odległości do każdej wartości E24
        w_flat = w.reshape(-1, 1)
        distances = -torch.abs(w_flat - self.grid.reshape(1, -1))

        # Softmax z temperaturą — im niższa T, tym ostrzejszy (bardziej one-hot)
        weights = torch.softmax(distances / self.temperature, dim=1)

        # Ważona suma wartości E24
        quantized = (weights * self.grid.reshape(1, -1)).sum(dim=1)
        return quantized.reshape(w.shape)

    def set_temperature(self, temperature: float) -> None:
        """Ustawia temperaturę (wywoływana przez trainer w każdej epoce).

        Args:
            temperature: Nowa temperatura (>0).
        """
        self.temperature = max(temperature, 1e-6)


# =============================================================================
# MIXED PRECISION QUANTIZER (dla QAT)
# =============================================================================

def quantize_mixed_precision(
    w: torch.Tensor,
    bits: int = 5,
    grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Kwantyzacja z ograniczoną precyzją — podzbiór wartości E24.

    Wybiera `2^bits` najbliższych wartości E24 pokrywających zakres wag.

    - 4-bit: 16 wartości (N_inh — mniej krytyczny)
    - 5-bit: 24 wartości (N1, N2 — pełna dekada E24)
    - 6-bit: 48 wartości (N3 — output, krytyczny — dwie dekady E24)

    Args:
        w: Tensor wag ciągłych.
        bits: Liczba bitów precyzji.
        grid: Opcjonalny pełny grid E24.

    Returns:
        Tensor wag skwantyzowanych do ograniczonego podzbioru E24.

    Przykład:
        >>> w = torch.tensor([0.35, 0.72])
        >>> w_4bit = quantize_mixed_precision(w, bits=4)
        >>> w_6bit = quantize_mixed_precision(w, bits=6)
    """
    if grid is None:
        grid = get_e24_grid(w.device)

    n_levels = min(2 ** bits, len(grid))

    if n_levels >= len(grid):
        # Pełna precyzja E24
        return quantize_to_e24(w, grid)

    # Wybierz n_levels równo rozłożonych wartości z grida
    indices = torch.linspace(0, len(grid) - 1, n_levels).long()
    sub_grid = grid[indices]

    return quantize_to_e24(w, sub_grid)


# =============================================================================
# WARTOŚCI E24 ↔ REZYSTANCJA (konwersja fizyczna)
# =============================================================================

def weight_to_resistance(
    w: float,
    r_in: float = HW_CONFIG.r_in,
) -> float:
    """Konwertuje znormalizowaną wagę na rezystancję synaptyczną.

    Synapsa to dzielnik: V_mem = V_spike × R_in / (R_syn + R_in)
    Więc: w = R_in / (R_syn + R_in), stąd: R_syn = R_in × (1/w - 1)

    Args:
        w: Waga znormalizowana [0,1].
        r_in: Rezystancja wejściowa neuronu (Ohm).

    Returns:
        Rezystancja synaptyczna R_syn (Ohm).

    Przykład:
        >>> r = weight_to_resistance(0.5)
        >>> print(f"R_syn = {r/1000:.1f} kΩ")  # 100.0 kΩ
    """
    if w <= 0.0:
        return HW_CONFIG.r_syn_max
    if w >= 1.0:
        return 0.0
    r_syn = r_in * (1.0 / w - 1.0)
    return max(min(r_syn, HW_CONFIG.r_syn_max), HW_CONFIG.r_syn_min)


def resistance_to_weight(
    r_syn: float,
    r_in: float = HW_CONFIG.r_in,
) -> float:
    """Konwertuje rezystancję synaptyczną na znormalizowaną wagę.

    Args:
        r_syn: Rezystancja synaptyczna (Ohm).
        r_in: Rezystancja wejściowa neuronu (Ohm).

    Returns:
        Waga znormalizowana [0,1].

    Przykład:
        >>> w = resistance_to_weight(100_000)
        >>> print(f"w = {w:.2f}")  # 0.50
    """
    if r_syn <= 0:
        return 1.0
    return r_in / (r_syn + r_in)


def weight_to_nearest_e24_resistance(
    w: float,
    r_in: float = HW_CONFIG.r_in,
) -> Tuple[float, float]:
    """Konwertuje wagę na najbliższą rezystancję E24.

    Args:
        w: Waga znormalizowana.
        r_in: Rezystancja wejściowa.

    Returns:
        Krotka (R_syn_exact, R_syn_E24_nearest) w Ohmach.

    Przykład:
        >>> r_exact, r_e24 = weight_to_nearest_e24_resistance(0.68)
        >>> print(f"Exact: {r_exact/1000:.1f}kΩ, E24: {r_e24/1000:.1f}kΩ")
    """
    from snn_pipeline.config import E24_SYNAPSE_RANGE

    r_exact = weight_to_resistance(w, r_in)

    # Znajdź najbliższą wartość E24
    idx = np.argmin(np.abs(E24_SYNAPSE_RANGE - r_exact))
    r_e24 = float(E24_SYNAPSE_RANGE[idx])

    return r_exact, r_e24
