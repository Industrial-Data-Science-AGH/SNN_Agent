# -*- coding: utf-8 -*-
"""
QAT Trainer — Quantization-Aware Training z mixed precision, calibration i fine-tuning.

Po HAT wykonuje dodatkowy fine-tuning z:
- Calibration dataset (50 próbek glass break → clip ranges)
- Per-synapse scale factors (trainable R_syn)
- Mixed precision: N3=6-bit, N1/N2=5-bit, N_inh=4-bit
- Fake quantization (E24 STE) podczas forward pass
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from snn_pipeline.config import (
    BASELINE_NEURONS,
    DEVICE,
    HW_CONFIG,
    PATH_CONFIG,
    TRAIN_CONFIG,
)
from snn_pipeline.losses import HardwareAwareLoss
from snn_pipeline.metrics import all_metrics, precision_score, recall_score
from snn_pipeline.snn_model import GlassBreakSNN


class QATTrainer:
    """Quantization-Aware Training trainer.

    Etap po HAT — fine-tuning z fake quantization przy mixed precision.

    Krok 1: Calibration — bierze 50 próbek glass_break, zapisuje zakresy aktywacji.
    Krok 2: Fine-tuning z fake quantization (STE) i per-synapse scale factors.
    Krok 3: Finalna kwantyzacja do E24 i zamrożenie.

    Attributes:
        model: Model SNN (po HAT).
        clip_ranges: Zakresy aktywacji z calibration (percentyle 0.1%/99.9%).
        history: Historia metryk per epoka QAT.

    Przykład:
        >>> qat = QATTrainer(model_after_hat)
        >>> qat.calibrate(train_loader)
        >>> qat.train(train_loader, val_loader, epochs=20)
    """

    def __init__(
        self,
        model: GlassBreakSNN,
        learning_rate: float = TRAIN_CONFIG.learning_rate * 0.1,  # Mniejszy LR dla fine-tuning
        config: Any = TRAIN_CONFIG,
    ) -> None:
        self.model = model.to(DEVICE)
        self.config = config
        self.criterion = HardwareAwareLoss(
            lambda_recall=config.lambda_recall,
            lambda_hw=config.lambda_hw * 2.0,  # Większa waga E24 w QAT
            recall_target=config.recall_target,
        )

        # Optymalizuj wagi + scale factors
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Clip ranges z calibration
        self.clip_ranges: Dict[str, Tuple[float, float]] = {}

        # Historia
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_recall": [],
            "val_loss": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

    @torch.no_grad()
    def calibrate(
        self,
        train_loader: DataLoader,
        n_samples: int = 50,
    ) -> Dict[str, Tuple[float, float]]:
        """Calibration — uruchamia model na próbkach glass_break i zapisuje zakresy aktywacji.

        Zbiera percentyle 0.1% i 99.9% aktywacji dla każdej warstwy.

        Args:
            train_loader: DataLoader (zbiór treningowy).
            n_samples: Liczba próbek kalibracyjnych.

        Returns:
            Słownik clip ranges {warstwa: (min, max)}.

        Przykład:
            >>> ranges = qat.calibrate(train_loader, n_samples=50)
            >>> print(ranges)
        """
        self.model.eval()

        # Zbierz aktywacje z glass_break samples
        activations: Dict[str, List[float]] = {
            "N1": [], "N2": [], "N3": [], "N_inh": [],
        }
        collected = 0

        for batch_spikes, batch_labels in train_loader:
            batch_spikes = batch_spikes.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            # Filtruj tylko pozytywne (glass_break)
            positive_mask = (batch_labels.flatten() == 1)
            if positive_mask.sum() == 0:
                continue

            pos_spikes = batch_spikes[positive_mask]
            trigger, neuron_spikes = self.model(pos_spikes)

            for name in activations:
                spikes = neuron_spikes[name]
                # Spike rate per sample
                rates = spikes.mean(dim=1).cpu().numpy()
                activations[name].extend(rates.tolist())

            collected += positive_mask.sum().item()
            if collected >= n_samples:
                break

        # Oblicz percentyle 0.1% i 99.9%
        for name, values in activations.items():
            if values:
                arr = np.array(values)
                low = float(np.percentile(arr, 0.1))
                high = float(np.percentile(arr, 99.9))
                self.clip_ranges[name] = (low, high)
            else:
                self.clip_ranges[name] = (0.0, 1.0)

        print(f"[QAT Calibration] Zebrano {collected} próbek glass_break.")
        for name, (lo, hi) in self.clip_ranges.items():
            print(f"  {name}: clip range = [{lo:.4f}, {hi:.4f}]")

        return self.clip_ranges

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Jeden epoch QAT fine-tuning.

        Args:
            train_loader: DataLoader treningowy.

        Returns:
            Metryki epoki.
        """
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        model_weights = [
            self.model.w_n1,
            self.model.w_n2,
            self.model.w_n3_from_n1,
            self.model.w_n3_from_n2,
            self.model.w_inh,
        ]

        for batch_spikes, batch_labels in train_loader:
            batch_spikes = batch_spikes.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            self.optimizer.zero_grad()
            trigger, _ = self.model(batch_spikes)

            loss, components = self.criterion(
                trigger, batch_labels, model_weights=model_weights
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.model.clamp_weights()

            total_loss += loss.item() * batch_spikes.size(0)
            all_preds.append(trigger.detach())
            all_targets.append(batch_labels.detach())

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        n = len(all_preds)

        return {
            "loss": total_loss / max(n, 1),
            "recall": recall_score(all_preds, all_targets),
        }

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Ewaluacja na zbiorze walidacyjnym."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_spikes, batch_labels in val_loader:
            batch_spikes = batch_spikes.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            trigger, _ = self.model(batch_spikes)
            loss, _ = self.criterion(trigger, batch_labels)
            total_loss += loss.item() * batch_spikes.size(0)
            all_preds.append(trigger)
            all_targets.append(batch_labels)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        n = len(all_preds)
        metrics = all_metrics(all_preds, all_targets)

        return {
            "loss": total_loss / max(n, 1),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Pętla treningowa QAT.

        Args:
            train_loader: DataLoader treningowy.
            val_loader: DataLoader walidacyjny.
            epochs: Liczba epok QAT.
            checkpoint_dir: Katalog na checkpointy.

        Returns:
            Historia metryk.

        Przykład:
            >>> history = qat.train(train_loader, val_loader, epochs=20)
        """
        if epochs is None:
            epochs = self.config.qat_epochs
        if checkpoint_dir is None:
            checkpoint_dir = PATH_CONFIG.checkpoint_dir

        # Włącz QAT mode (mixed precision)
        self.model.set_quantize_mode("qat")
        self.model.enable_mismatch(True)

        best_f1 = 0.0

        print("=" * 70)
        print("  QUANTIZATION-AWARE TRAINING (QAT)")
        print(f"  Epochs: {epochs}, LR: {self.optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Mixed precision: N3=6bit, N1/N2=5bit, N_inh=4bit")
        print("=" * 70)

        for epoch in tqdm(range(epochs), desc="QAT training"):
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)

            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_recall"].append(train_metrics["recall"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_f1"].append(val_metrics["f1"])

            if epoch % 5 == 0 or epoch == epochs - 1:
                tqdm.write(
                    f"  QAT Epoch {epoch:3d}/{epochs} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Val P/R/F1: {val_metrics['precision']:.2f}/{val_metrics['recall']:.2f}/{val_metrics['f1']:.2f}"
                )

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_metrics": val_metrics,
                }, checkpoint_dir / "qat_best.pt")

        # Przełącz na twardą kwantyzację
        self.model.set_quantize_mode("hat")
        self.model.enable_mismatch(False)

        print(f"\n[QAT DONE] Najlepsze F1: {best_f1:.2f}")
        return self.history

    def plot_qat_curves(self, save_path: Optional[str] = None) -> None:
        """Rysuje learning curves QAT.

        Args:
            save_path: Ścieżka do zapisu wykresu.
        """
        if save_path is None:
            save_path = str(PATH_CONFIG.output_dir / "qat_learning_curves.png")

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("QAT Learning Curves", fontsize=14, fontweight="bold")

        epochs = self.history["epoch"]

        axes[0].plot(epochs, self.history["train_loss"], label="Train", color="blue")
        axes[0].plot(epochs, self.history["val_loss"], label="Val", color="orange")
        axes[0].set_title("Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs, self.history["val_recall"], label="Recall", color="green")
        axes[1].plot(epochs, self.history["val_precision"], label="Precision", color="red")
        axes[1].set_title("Precision / Recall")
        axes[1].set_ylim(-0.05, 1.05)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(epochs, self.history["val_f1"], label="F1", color="purple")
        axes[2].set_title("F1 Score")
        axes[2].set_ylim(-0.05, 1.05)
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] QAT curves saved to {save_path}")
