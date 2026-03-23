# -*- coding: utf-8 -*-
"""
Funkcje straty — hardware-aware loss function dla treningu HAT i QAT.

FIX v2: Naprawiony degenerate collapse — dodano precision penalty i F1 penalty.
Stary loss karał TYLKO za niski recall → gradient zabijał inhibitor.
Nowy loss balansuje recall i precision przez F1 + osobne penalty.

L_total = L_BCE + λ_f1 × (1 - soft_F1)² + λ_precision × max(0, p_target - p)²
          + λ_recall × max(0, r_target - r)² + λ_hw × Σ|w - quant(w)|²
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from snn_pipeline.config import TRAIN_CONFIG
from snn_pipeline.e24_quantizer import quantize_to_e24


class HardwareAwareLoss(nn.Module):
    """Hardware-aware loss z balansem precision/recall (F1-based).

    Naprawiony problem degenerate collapse: stara wersja miała TYLKO recall
    penalty co powodowało że gradient zabijał inhibitor (najtańszy sposób
    na recall=1.0 = "strzelaj na wszystko").

    Nowa wersja:
    - F1 penalty jako główny driver (balansuje P i R)
    - Osobny precision penalty (chroni przed false positives)
    - Mniejszy recall penalty (nie dominuje)
    - E24 regularization (bez zmian)

    Attributes:
        lambda_f1: Waga kary za niski F1 (domyślnie 2.0).
        lambda_precision: Waga kary za niski precision (domyślnie 1.5).
        lambda_recall: Waga kary za niski recall (domyślnie 0.5).
        lambda_hw: Waga regularyzacji E24 (domyślnie 0.3).
        precision_target: Docelowy precision (domyślnie 0.90).
        recall_target: Docelowy recall (domyślnie 0.85).

    Przykład:
        >>> criterion = HardwareAwareLoss()
        >>> pred = torch.tensor([[0.8], [0.1], [0.3]])
        >>> tgt = torch.tensor([[1.0], [0.0], [1.0]])
        >>> loss, comps = criterion(pred, tgt, [torch.tensor([0.35])])
        >>> print(f"Total: {comps['total']:.4f}")
    """

    def __init__(
        self,
        lambda_f1: float = 2.0,
        lambda_precision: float = 1.5,
        lambda_recall: float = 0.5,
        lambda_hw: float = 0.3,
        precision_target: float = 0.90,
        recall_target: float = TRAIN_CONFIG.recall_target,
    ) -> None:
        super().__init__()
        self.lambda_f1 = lambda_f1
        self.lambda_precision = lambda_precision
        self.lambda_recall = lambda_recall
        self.lambda_hw = lambda_hw
        self.precision_target = precision_target
        self.recall_target = recall_target
        self.bce = nn.BCELoss()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        model_weights: Optional[list] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Oblicza hardware-aware loss z balansem precision/recall.

        Args:
            predictions: Predykcje modelu (batch, 1), wartości [0,1].
            targets: Etykiety (batch, 1), wartości {0, 1}.
            model_weights: Lista tensorów wag modelu (do regularyzacji E24).

        Returns:
            Krotka (total_loss, components_dict).
        """
        # Clamp predictions do bezpiecznego zakresu dla BCE
        pred_safe = torch.clamp(predictions, 1e-7, 1.0 - 1e-7)

        # 1. BCE (Binary Cross Entropy) — główny loss klasyfikacyjny
        loss_bce = self.bce(pred_safe, targets)

        # 2. Soft F1 penalty — NOWY, balansuje P i R jednocześnie
        soft_p, soft_r, soft_f1 = self._soft_precision_recall_f1(pred_safe, targets)
        loss_f1 = (1.0 - soft_f1) ** 2

        # 3. Precision penalty — NOWY, chroni przed false positives
        precision_deficit = torch.clamp(self.precision_target - soft_p, min=0.0)
        loss_precision = precision_deficit ** 2

        # 4. Recall penalty — zmniejszona waga (0.5 zamiast 2.0)
        recall_deficit = torch.clamp(self.recall_target - soft_r, min=0.0)
        loss_recall = recall_deficit ** 2

        # 5. E24 regularization — bez zmian
        loss_hw = torch.tensor(0.0, device=predictions.device)
        if model_weights is not None:
            for w in model_weights:
                w_quantized = quantize_to_e24(w.detach())
                loss_hw = loss_hw + torch.sum((w - w_quantized) ** 2)

        # Total loss z nowymi wagami
        total = (
            loss_bce
            + self.lambda_f1 * loss_f1
            + self.lambda_precision * loss_precision
            + self.lambda_recall * loss_recall
            + self.lambda_hw * loss_hw
        )

        components = {
            "bce": loss_bce.item(),
            "f1_penalty": loss_f1.item(),
            "precision_penalty": loss_precision.item(),
            "recall_penalty": loss_recall.item(),
            "hw_regularization": loss_hw.item(),
            "soft_precision": soft_p.item(),
            "soft_recall": soft_r.item(),
            "soft_f1": soft_f1.item(),
            "total": total.item(),
        }

        return total, components

    def _soft_precision_recall_f1(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Oblicza differentiable precision, recall i F1.

        Używa soft (ciągłych) wersji TP, FP, FN żeby gradient mógł przepływać.

        soft_TP = Σ (pred × target)
        soft_FP = Σ (pred × (1 - target))
        soft_FN = Σ ((1 - pred) × target)
        soft_precision = TP / (TP + FP + ε)
        soft_recall = TP / (TP + FN + ε)
        soft_F1 = 2 × P × R / (P + R + ε)

        Args:
            predictions: Predykcje [0,1].
            targets: Etykiety {0,1}.

        Returns:
            Krotka (soft_precision, soft_recall, soft_f1).
        """
        eps = 1e-8

        soft_tp = torch.sum(predictions * targets)
        soft_fp = torch.sum(predictions * (1.0 - targets))
        soft_fn = torch.sum((1.0 - predictions) * targets)

        soft_precision = soft_tp / (soft_tp + soft_fp + eps)
        soft_recall = soft_tp / (soft_tp + soft_fn + eps)
        soft_f1 = 2.0 * soft_precision * soft_recall / (soft_precision + soft_recall + eps)

        return soft_precision, soft_recall, soft_f1


class FocalLoss(nn.Module):
    """Focal Loss — extra opcja dla niezbalansowanego datasetu.

    FL(p) = -α × (1-p)^γ × log(p) dla klasy pozytywnej
    FL(p) = -(1-α) × p^γ × log(1-p) dla klasy negatywnej

    Przydatne gdy glass_break stanowi <5% datasetu.

    Attributes:
        alpha: Waga klasy pozytywnej (domyślnie 0.8 — faworyzuj glass_break).
        gamma: Parametr fokusu (domyślnie 2.0 — mocno tłumi łatwe przykłady).

    Przykład:
        >>> focal = FocalLoss(alpha=0.8, gamma=2.0)
        >>> loss = focal(torch.tensor([[0.9]]), torch.tensor([[1.0]]))
    """

    def __init__(self, alpha: float = 0.8, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Oblicza Focal Loss.

        Args:
            predictions: Predykcje [0,1].
            targets: Etykiety {0,1}.

        Returns:
            Focal loss (skalar).
        """
        pred_safe = torch.clamp(predictions, 1e-7, 1.0 - 1e-7)
        bce = F.binary_cross_entropy(pred_safe, targets, reduction='none')

        # p_t = prawdopodobieństwo poprawnej klasy
        p_t = pred_safe * targets + (1.0 - pred_safe) * (1.0 - targets)

        # Alpha weighting
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        # Focal weighting
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma

        loss = (focal_weight * bce).mean()
        return loss
