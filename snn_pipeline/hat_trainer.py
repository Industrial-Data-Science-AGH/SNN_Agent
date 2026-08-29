# -*- coding: utf-8 -*-
"""
HAT Trainer — Hardware-Aware Training z temperature annealing, mismatch simulation,
threshold sweep i Pareto front tracking.

FIX v2:
- DODANO: calibrate_thresholds() przed treningiem — obniża V_th do rozkładu danych
- ZMIENIONY: checkpoint po max F1 (nie max recall) — zapobiega zapisowi degenerate epoch
- ZMIENIONY: inicjalizacja z losses v2 (F1 + precision + recall penalty)
"""

import time
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
    DEVICE,
    PATH_CONFIG,
    TRAIN_CONFIG,
)
from snn_pipeline.e24_quantizer import quantize_to_e24_with_error
from snn_pipeline.losses import HardwareAwareLoss
from snn_pipeline.metrics import all_metrics, precision_score, recall_score, f1_score
from snn_pipeline.snn_model import GlassBreakSNN


class HATTrainer:
    """Hardware-Aware Training trainer (v2 — fixed degenerate collapse).

    Zmiany vs v1:
    - calibrate_thresholds() — kalibruje V_th do rozkładu aktywacji danych
    - Checkpoint po max F1 (nie max recall)
    - Loss z F1 + precision penalty (nie recall-only)

    Przykład:
        >>> model = GlassBreakSNN(quantize_mode="gumbel")
        >>> trainer = HATTrainer(model)
        >>> trainer.calibrate_thresholds(train_loader)   # FIX v2
        >>> trainer.train(train_loader, val_loader, epochs=50)
    """

    def __init__(
        self,
        model: GlassBreakSNN,
        learning_rate: float = TRAIN_CONFIG.learning_rate,
        config: Any = TRAIN_CONFIG,
    ) -> None:
        self.model = model.to(DEVICE)
        self.config = config

        # FIX v2: loss z F1 + precision penalty (nowy API HardwareAwareLoss)
        self.criterion = HardwareAwareLoss(
            lambda_f1=2.0,
            lambda_precision=1.5,
            lambda_recall=0.5,
            lambda_hw=0.3,
            precision_target=0.90,
            recall_target=config.recall_target,
        )
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=1e-5
        )

        # Historia metryk
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "train_precision": [],
            "train_recall": [],
            "train_f1": [],
            "val_loss": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            "temperature": [],
        }

        # Pareto front
        self.pareto_fronts: List[List[Tuple[float, float, float]]] = []

    # ==========================================================================
    # FIX v2: KALIBRACJA PROGÓW V_th PRZED TRENINGIEM
    # ==========================================================================
    @torch.no_grad()
    def calibrate_thresholds(
        self,
        train_loader: DataLoader,
        percentile: float = 10.0,
    ) -> None:
        """Kalibruje progi V_th neuronów do rozkładu aktywacji na danych treningowych.

        PROBLEM: Baseline V_th z symulacji (0.5-0.8) są za wysokie dla rzeczywistych
        danych ESC-50. Glass break ma peak energy ~0.04-0.06 → żaden spike nie przekracza
        progu → recall=0.0 od pierwszej epoki → gradient panikuje i zabija inhibitor.

        NAPRAWA: Uruchom model na danych, zbierz rozkład spike rate N1/N2/N3,
        ustaw V_th na percentyl 10% rozkładu. To daje sensowny punkt startowy
        (część próbek glass_break przechodzi, ale nie wszystkie → gradient ma
        na czym pracować w obie strony).

        Args:
            train_loader: DataLoader treningowy.
            percentile: Percentyl rozkładu spike rate do ustawienia V_th.

        Przykład:
            >>> trainer.calibrate_thresholds(train_loader, percentile=10)
        """
        self.model.eval()
        # Tymczasowo wyłącz kwantyzację i mismatch
        old_mode = self.model.quantize_mode
        old_mismatch = self.model.mismatch_enabled
        self.model.set_quantize_mode("none")
        self.model.enable_mismatch(False)

        # Zbierz spike rates per neuron
        rates: Dict[str, List[float]] = {"N1": [], "N2": [], "N3": [], "N_inh": []}

        for batch_spikes, batch_labels in train_loader:
            batch_spikes = batch_spikes.to(DEVICE)
            trigger, neuron_spikes = self.model(batch_spikes)

            for name in rates:
                sr = neuron_spikes[name].mean(dim=1).cpu().numpy()  # (batch,)
                rates[name].extend(sr.tolist())

        # Ustaw V_th = percentyl rozkładu spike rate
        print(f"\n[V_th Calibration] Percentyl {percentile}% z rozkładu spike rate:")
        for name, values in rates.items():
            if not values:
                continue
            arr = np.array(values)
            # Statystyki
            p10 = np.percentile(arr, percentile) if len(arr) > 0 else 0.1
            p50 = np.percentile(arr, 50) if len(arr) > 0 else 0.5
            p90 = np.percentile(arr, 90) if len(arr) > 0 else 0.9

            # Nowy próg = percentyl + mały margines
            # Clamp żeby nie był zbyt niski (min 0.01) lub zbyt wysoki (max 0.5)
            new_vth = max(0.01, min(float(p10) + 0.005, 0.5))

            # Ustaw w modelu
            vth_param_name = f"vth_{name.lower()}"
            if name == "N_inh":
                vth_param_name = "vth_inh"
            param = getattr(self.model, vth_param_name, None)
            if param is not None:
                old_vth = param.item()
                param.data.fill_(new_vth)
                # Aktualizuj też snntorch LIF threshold
                lif_name = f"lif_{name.lower()}"
                if name == "N_inh":
                    lif_name = "lif_inh"
                lif = getattr(self.model, lif_name, None)
                if lif is not None:
                    lif.threshold = torch.tensor(new_vth)

                print(f"  {name}: spike_rate p10={p10:.4f} p50={p50:.4f} p90={p90:.4f} "
                      f"-> V_th: {old_vth:.3f} -> {new_vth:.3f}")

        # Przywróć tryb
        self.model.set_quantize_mode(old_mode)
        self.model.enable_mismatch(old_mismatch)
        print()

    def _get_temperature(self, epoch: int, total_epochs: int) -> float:
        """Temperatura Gumbel-softmax: exponential decay T_start -> T_end."""
        ratio = epoch / max(total_epochs - 1, 1)
        temp = self.config.temp_start * (
            (self.config.temp_end / self.config.temp_start) ** ratio
        )
        return max(temp, self.config.temp_end)

    def _get_model_weights(self) -> list:
        """Wagi modelu do regularyzacji E24."""
        return [
            self.model.w_n1,
            self.model.w_n2,
            self.model.w_n3_from_n1,
            self.model.w_n3_from_n2,
            self.model.w_inh,
        ]

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Jeden epoch treningu."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch_spikes, batch_labels in train_loader:
            batch_spikes = batch_spikes.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            self.optimizer.zero_grad()
            trigger, _ = self.model(batch_spikes)

            loss, components = self.criterion(
                trigger, batch_labels, model_weights=self._get_model_weights()
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
        n_samples = len(all_preds)

        return {
            "loss": total_loss / max(n_samples, 1),
            "precision": precision_score(all_preds, all_targets),
            "recall": recall_score(all_preds, all_targets),
            "f1": f1_score(all_preds, all_targets),
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
            loss, _ = self.criterion(trigger, batch_labels, self._get_model_weights())

            total_loss += loss.item() * batch_spikes.size(0)
            all_preds.append(trigger)
            all_targets.append(batch_labels)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        n_samples = len(all_preds)
        metrics = all_metrics(all_preds, all_targets)

        return {
            "loss": total_loss / max(n_samples, 1),
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        }

    @torch.no_grad()
    def _threshold_sweep(
        self, val_loader: DataLoader
    ) -> List[Tuple[float, float, float]]:
        """Sweep V_th decision threshold ∈ [0.0, 0.5] i mierzy P/R."""
        self.model.eval()

        all_preds = []
        all_targets = []
        for batch_spikes, batch_labels in val_loader:
            batch_spikes = batch_spikes.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)
            trigger, _ = self.model(batch_spikes)
            all_preds.append(trigger)
            all_targets.append(batch_labels)

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        pareto_points = []
        # FIX v2: zakres sweep dostosowany do spike rate (0.0-0.5, nie 0.3-1.5)
        for threshold in np.arange(0.0, 0.55, 0.01):
            p = precision_score(all_preds, all_targets, threshold=threshold)
            r = recall_score(all_preds, all_targets, threshold=threshold)
            pareto_points.append((float(threshold), p, r))

        return pareto_points

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
        checkpoint_dir: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Główna pętla treningowa HAT (v2 — fixed).

        FIX v2: Checkpoint po max F1 (nie max recall).

        Args:
            train_loader: DataLoader treningowy.
            val_loader: DataLoader walidacyjny.
            epochs: Liczba epok.
            checkpoint_dir: Katalog na checkpointy.

        Returns:
            Historia metryk.
        """
        if epochs is None:
            epochs = self.config.hat_epochs
        if checkpoint_dir is None:
            checkpoint_dir = PATH_CONFIG.checkpoint_dir

        # Włącz mismatch simulation i Gumbel quantization
        self.model.set_quantize_mode("gumbel")
        self.model.enable_mismatch(True)

        # FIX v2: checkpoint po F1, nie recall
        best_f1 = -1.0
        best_epoch = 0

        print("=" * 70)
        print("  HARDWARE-AWARE TRAINING (HAT) v2 — Fixed")
        print(f"  Epochs: {epochs}, LR: {self.optimizer.param_groups[0]['lr']}")
        print(f"  Loss: BCE + F1 penalty + precision penalty + recall penalty + E24")
        print(f"  Temperature: {self.config.temp_start} -> {self.config.temp_end}")
        print(f"  Mismatch: ±{self.config.mismatch_weight_pct}% wag, ±{self.config.mismatch_vth_mv}mV V_th")
        print(f"  Checkpoint: max F1 (nie max recall)")
        print("=" * 70)

        for epoch in tqdm(range(epochs), desc="HAT Training"):
            # 1. Temperatura Gumbel-softmax
            temp = self._get_temperature(epoch, epochs)
            self.model.gumbel_quantizer.set_temperature(temp)

            # 2. Train
            train_metrics = self._train_epoch(train_loader)

            # 3. Validate
            val_metrics = self._validate(val_loader)

            # 4. Threshold sweep
            pareto = self._threshold_sweep(val_loader)
            self.pareto_fronts.append(pareto)

            # 5. Logowanie
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_precision"].append(train_metrics["precision"])
            self.history["train_recall"].append(train_metrics["recall"])
            self.history["train_f1"].append(train_metrics["f1"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_precision"].append(val_metrics["precision"])
            self.history["val_recall"].append(val_metrics["recall"])
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["temperature"].append(temp)

            # 6. Progress co 5 epok
            if epoch % 5 == 0 or epoch == epochs - 1:
                # Log inhibitor health
                w_inh = self.model.w_inh.item()
                w_inh_n3 = self.model.w_inh_to_n3.item()
                tqdm.write(
                    f"  Epoch {epoch:3d}/{epochs} | "
                    f"T={temp:.2f} | "
                    f"Loss: {train_metrics['loss']:.4f} | "
                    f"Val P/R/F1: {val_metrics['precision']:.2f}/{val_metrics['recall']:.2f}/{val_metrics['f1']:.2f} | "
                    f"inh={w_inh:.3f}/{w_inh_n3:.3f}"
                )

            # 7. FIX v2: Checkpoint po max F1 (nie max recall)
            current_f1 = val_metrics["f1"]
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "weights": self.model.get_weights_dict(),
                    "thresholds": self.model.get_thresholds_dict(),
                }, checkpoint_dir / "hat_best.pt")

        # Załaduj najlepszy checkpoint (po F1)
        best_ckpt = checkpoint_dir / "hat_best.pt"
        if best_ckpt.exists() and best_f1 > 0:
            loaded = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
            self.model.load_state_dict(loaded["model_state_dict"])
            print(f"\n[HAT] Załadowano najlepszy checkpoint z epoki {best_epoch} (F1={best_f1:.3f})")

        # Na koniec przełącz na twardą kwantyzację E24
        self.model.set_quantize_mode("hat")
        self.model.enable_mismatch(False)

        print(f"[HAT DONE] Najlepsze F1: {best_f1:.3f} (epoch {best_epoch})")
        print(f"[HAT DONE] Wagi: {self.model.get_weights_dict()}")
        print(f"[HAT DONE] Progi: {self.model.get_thresholds_dict()}")

        return self.history

    def plot_learning_curves(self, save_path: Optional[str] = None) -> None:
        """Rysuje learning curves: loss, recall, precision, F1, temperature."""
        if save_path is None:
            save_path = str(PATH_CONFIG.output_dir / "hat_learning_curves.png")

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("HAT Learning Curves (v2 — Fixed)", fontsize=14, fontweight="bold")

        epochs = self.history["epoch"]

        # Loss
        axes[0, 0].plot(epochs, self.history["train_loss"], label="Train", color="blue")
        axes[0, 0].plot(epochs, self.history["val_loss"], label="Val", color="orange")
        axes[0, 0].set_title("Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # F1 + Recall
        axes[0, 1].plot(epochs, self.history["val_f1"], label="Val F1", color="purple", linewidth=2)
        axes[0, 1].plot(epochs, self.history["val_recall"], label="Val Recall", color="green", linestyle="--")
        axes[0, 1].plot(epochs, self.history["val_precision"], label="Val Precision", color="red", linestyle="--")
        axes[0, 1].axhline(y=self.config.recall_target, color="gray", linestyle=":",
                           label=f"Recall target ({self.config.recall_target})")
        axes[0, 1].set_title("F1 / Precision / Recall")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylim(-0.05, 1.05)
        axes[0, 1].legend(fontsize=8)
        axes[0, 1].grid(True, alpha=0.3)

        # Train metrics
        axes[1, 0].plot(epochs, self.history["train_f1"], label="Train F1", color="purple")
        axes[1, 0].plot(epochs, self.history["train_recall"], label="Train Recall", color="green", linestyle="--")
        axes[1, 0].plot(epochs, self.history["train_precision"], label="Train Precision", color="red", linestyle="--")
        axes[1, 0].set_title("Train Metrics")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylim(-0.05, 1.05)
        axes[1, 0].legend(fontsize=8)
        axes[1, 0].grid(True, alpha=0.3)

        # Temperature
        axes[1, 1].plot(epochs, self.history["temperature"], color="teal")
        axes[1, 1].set_title("Gumbel-Softmax Temperature")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_yscale("log")
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[INFO] Learning curves saved to {save_path}")

    def get_weight_table(self) -> str:
        """Tabela wag: oryginalna -> trened. -> E24 -> błąd%."""
        from snn_pipeline.config import BASELINE_NEURONS

        weights = self.model.get_weights_dict()
        lines = [
            "=" * 80,
            f"{'Neuron':<12} {'Synapsa':<18} {'Baseline':<10} {'Trened.':<10} {'E24':<10} {'Błąd %':<8}",
            "-" * 80,
        ]

        baseline_map = {
            "w_n1": ("N1", BASELINE_NEURONS['N1'].w_in[0]),
            "w_n2": ("N2", BASELINE_NEURONS['N2'].w_in[0]),
            "w_n3_from_n1": ("N3", BASELINE_NEURONS['N3'].w_in[0]),
            "w_n3_from_n2": ("N3", BASELINE_NEURONS['N3'].w_in[1]),
            "w_inh": ("N_inh", BASELINE_NEURONS['N_inh'].w_in[0]),
            "w_inh_to_n3": ("N_inh->N3", -0.5),
        }

        for name, w_val in weights.items():
            w_tensor = torch.tensor([abs(w_val)])
            w_q, err_abs, err_pct = quantize_to_e24_with_error(w_tensor)
            neuron, orig = baseline_map.get(name, ("?", 0.0))

            sign = "-" if w_val < 0 else ""
            lines.append(
                f"{neuron:<12} {name:<18} {orig:<10.3f} {sign}{abs(w_val):<9.3f} "
                f"{sign}{w_q.item():<9.3f} {err_pct.item():<8.1f}"
            )

        lines.append("=" * 80)
        return "\n".join(lines)
