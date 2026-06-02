# -*- coding: utf-8 -*-
"""
Model SNN — architektura sieci Spiking Neural Network na bazie snntorch.

Implementuje 4 neurony LIF:
- N1: detektor wysokiej częstotliwości (>2kHz) — reaguje na ostre transients szkła
- N2: detektor temporal (burst pattern) — rozpoznaje krótki, impulsowy charakter
- N3: neuron wyjściowy — zbiera N1+N2, gdy oba aktywne → TRIGGER
- N_inh: neuron hamujący — reaguje na ciągłe szumy (HVAC <500Hz), tłumi N1-N3

Wagi synaptyczne mogą być kwantyzowane do E24 (HAT) lub mixed precision (QAT)
poprzez fake quantization z STE.
"""

import torch
import torch.nn as nn
import snntorch as snn
from typing import Dict, Optional, Tuple

from snn_pipeline.config import (
    BASELINE_NEURONS,
    DEVICE,
    HW_CONFIG,
    LIF_CONFIG,
    TRAIN_CONFIG,
)
from snn_pipeline.e24_quantizer import (
    E24GumbelQuantizer,
    e24_ste,
    quantize_mixed_precision,
    get_e24_grid,
)


class GlassBreakSNN(nn.Module):
    """Spiking Neural Network do detekcji rozbijanego szkła.

    Architektura:
    ```
    Spike Input (kanały energii w pasmach częstotliwości)
        ├─→ N1 (HF detektor, beta=decay) ──┐
        ├─→ N2 (temporal detektor)  ────────┤
        └─→ N_inh (inhibitor <500Hz) ──────┤ (hamujący)
                                            ↓
                                    N3 (output) → TRIGGER
    ```

    Wagi:
    - w_n1: waga synapsy wejście→N1
    - w_n2: waga synapsy wejście→N2
    - w_n3_from_n1: waga synapsy N1→N3
    - w_n3_from_n2: waga synapsy N2→N3
    - w_inh: waga synapsy wejście→N_inh
    - w_inh_to_n3: waga hamowania N_inh→N3 (ujemna)

    Progi V_th:
    - vth_n1, vth_n2, vth_n3, vth_inh — treninowalne progi wyzwalania

    Attributes:
        quantize_mode: "none", "hat" (E24 STE), "qat" (mixed precision), "gumbel"
        mismatch_enabled: Czy dodawać szum mismatch (±1% wag, ±5mV V_th)
        gumbel_quantizer: Kwantyzator Gumbel-softmax (używany w trybie "gumbel")

    Przykład:
        >>> model = GlassBreakSNN()
        >>> spikes_in = torch.randn(32, 1, 100)  # batch=32, 1 kanał, 100 timesteps
        >>> trigger, neuron_spikes = model(spikes_in)
        >>> print(f"Trigger shape: {trigger.shape}")  # (32, 1)
    """

    def __init__(
        self,
        beta: Optional[float] = None,
        quantize_mode: str = "none",
    ) -> None:
        super().__init__()

        if beta is None:
            beta = LIF_CONFIG.beta

        # =====================================================================
        # WAGI SYNAPTYCZNE — trainable parameters
        # =====================================================================
        baseline = BASELINE_NEURONS

        # Wejście → N1 (detektor HF)
        self.w_n1 = nn.Parameter(torch.tensor([baseline["N1"].w_in[0]]))
        # Wejście → N2 (detektor temporal)
        self.w_n2 = nn.Parameter(torch.tensor([baseline["N2"].w_in[0]]))
        # N1 → N3
        self.w_n3_from_n1 = nn.Parameter(torch.tensor([baseline["N3"].w_in[0]]))
        # N2 → N3
        self.w_n3_from_n2 = nn.Parameter(torch.tensor([baseline["N3"].w_in[1]]))
        # Wejście → N_inh (inhibitor)
        self.w_inh = nn.Parameter(torch.tensor([baseline["N_inh"].w_in[0]]))
        # N_inh → N3 (hamowanie — ujemna waga)
        self.w_inh_to_n3 = nn.Parameter(torch.tensor([-0.5]))

        # =====================================================================
        # PROGI V_th — trainable
        # =====================================================================
        self.vth_n1 = nn.Parameter(torch.tensor([baseline["N1"].v_th]))
        self.vth_n2 = nn.Parameter(torch.tensor([baseline["N2"].v_th]))
        self.vth_n3 = nn.Parameter(torch.tensor([baseline["N3"].v_th]))
        self.vth_inh = nn.Parameter(torch.tensor([baseline["N_inh"].v_th]))

        # =====================================================================
        # NEURONY LIF (snntorch)
        # =====================================================================
        # Beta = decay rate membrany = exp(-dt/tau_m)
        self.lif_n1 = snn.Leaky(beta=beta, threshold=baseline["N1"].v_th,
                                 learn_beta=False, learn_threshold=True)
        self.lif_n2 = snn.Leaky(beta=beta, threshold=baseline["N2"].v_th,
                                 learn_beta=False, learn_threshold=True)
        self.lif_n3 = snn.Leaky(beta=beta, threshold=baseline["N3"].v_th,
                                 learn_beta=False, learn_threshold=True)
        self.lif_inh = snn.Leaky(beta=beta, threshold=baseline["N_inh"].v_th,
                                  learn_beta=False, learn_threshold=True)

        # =====================================================================
        # KONFIGURACJA KWANTYZACJI
        # =====================================================================
        self.quantize_mode = quantize_mode
        self.mismatch_enabled = False

        # Gumbel-softmax quantizer (osobny bo ma temperaturę)
        self.gumbel_quantizer = E24GumbelQuantizer(temperature=TRAIN_CONFIG.temp_start)

        # QAT — per-synapse scale factors (trainable)
        self.scale_n1 = nn.Parameter(torch.tensor([100_000.0]))
        self.scale_n2 = nn.Parameter(torch.tensor([100_000.0]))
        self.scale_n3_1 = nn.Parameter(torch.tensor([100_000.0]))
        self.scale_n3_2 = nn.Parameter(torch.tensor([100_000.0]))
        self.scale_inh = nn.Parameter(torch.tensor([100_000.0]))

    def _apply_quantization(self, w: torch.Tensor, bits: int = 5) -> torch.Tensor:
        """Stosuje kwantyzację wag zależnie od trybu.

        Args:
            w: Tensor wag ciągłych.
            bits: Precyzja (dla QAT mixed precision).

        Returns:
            Tensor wag (ewentualnie skwantyzowanych).
        """
        if self.quantize_mode == "none":
            return w
        elif self.quantize_mode == "hat":
            return e24_ste(w)
        elif self.quantize_mode == "gumbel":
            return self.gumbel_quantizer(w)
        elif self.quantize_mode == "qat":
            return quantize_mixed_precision(w, bits=bits)
        return w

    def _apply_mismatch(
        self,
        w: torch.Tensor,
        vth: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Dodaje szum mismatch (symulacja tolerancji rezystorów i dryfu V_th).

        Args:
            w: Waga synaptyczna.
            vth: Próg V_th.

        Returns:
            Krotka (w_noisy, vth_noisy).
        """
        if not self.mismatch_enabled or not self.training:
            return w, vth

        # ±1% szum na wagach (tolerancja rezystorów)
        w_noise = torch.randn_like(w) * (TRAIN_CONFIG.mismatch_weight_pct / 100.0)
        w_noisy = w * (1.0 + w_noise)

        # ±5mV szum na V_th (dryf termiczny) — znormalizowane do [0,1]
        vth_noise_normalized = TRAIN_CONFIG.mismatch_vth_mv / 1000.0 / HW_CONFIG.vcc
        vth_noise = torch.randn_like(vth) * vth_noise_normalized
        vth_noisy = vth + vth_noise

        return w_noisy, vth_noisy

    def forward(
        self,
        spike_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass przez sieć SNN.

        Symuluje krok po kroku propagację spike'ów przez 4 neurony.

        Args:
            spike_input: Tensor spike'ów wejściowych, kształt (batch, channels, timesteps).

        Returns:
            Krotka:
              - trigger: Tensor (batch, 1) — prawdopodobieństwo triggera (mean spike rate N3).
              - neuron_spikes: Słownik z tensorami spike'ów każdego neuronu.

        Przykład:
            >>> model = GlassBreakSNN()
            >>> spikes = torch.randn(8, 1, 100)
            >>> trigger, info = model(spikes)
        """
        batch_size = spike_input.shape[0]
        n_timesteps = spike_input.shape[-1]

        # Jeśli wejście ma wiele kanałów, uśrednij (simplification)
        if spike_input.dim() == 3 and spike_input.shape[1] > 1:
            x = spike_input.mean(dim=1)  # (batch, timesteps)
        elif spike_input.dim() == 3:
            x = spike_input.squeeze(1)  # (batch, timesteps)
        else:
            x = spike_input  # (batch, timesteps)

        # =====================================================================
        # KWANTYZACJA WAG (qat_bits per neuron)
        # =====================================================================
        w_n1 = self._apply_quantization(self.w_n1, bits=BASELINE_NEURONS["N1"].qat_bits)
        w_n2 = self._apply_quantization(self.w_n2, bits=BASELINE_NEURONS["N2"].qat_bits)
        w_n3_1 = self._apply_quantization(self.w_n3_from_n1, bits=BASELINE_NEURONS["N3"].qat_bits)
        w_n3_2 = self._apply_quantization(self.w_n3_from_n2, bits=BASELINE_NEURONS["N3"].qat_bits)
        w_inh = self._apply_quantization(self.w_inh, bits=BASELINE_NEURONS["N_inh"].qat_bits)
        w_inh_n3 = self.w_inh_to_n3  # Waga hamująca — nie kwantyzujemy (ma być ujemna)

        # =====================================================================
        # MISMATCH (szum hardware)
        # =====================================================================
        w_n1, vth_n1 = self._apply_mismatch(w_n1, self.vth_n1)
        w_n2, vth_n2 = self._apply_mismatch(w_n2, self.vth_n2)
        w_n3_1, vth_n3 = self._apply_mismatch(w_n3_1, self.vth_n3)
        w_n3_2, _ = self._apply_mismatch(w_n3_2, self.vth_n3)
        w_inh, vth_inh = self._apply_mismatch(w_inh, self.vth_inh)

        # =====================================================================
        # INICJALIZACJA STANÓW NEURONÓW
        # =====================================================================
        mem_n1 = self.lif_n1.init_leaky()
        mem_n2 = self.lif_n2.init_leaky()
        mem_n3 = self.lif_n3.init_leaky()
        mem_inh = self.lif_inh.init_leaky()

        # Zbieracze spike'ów
        spk_n1_rec = []
        spk_n2_rec = []
        spk_n3_rec = []
        spk_inh_rec = []

        # =====================================================================
        # PĘTLA CZASOWA
        # =====================================================================
        for t in range(n_timesteps):
            x_t = x[:, t]  # (batch,)

            # --- Warstwa 1: N1, N2, N_inh (równoległa) ---
            # Prąd synaptyczny = wejście × waga
            cur_n1 = x_t * w_n1       # (batch,)
            cur_n2 = x_t * w_n2       # (batch,)
            cur_inh = x_t * w_inh     # (batch,)

            spk_n1, mem_n1 = self.lif_n1(cur_n1, mem_n1)
            spk_n2, mem_n2 = self.lif_n2(cur_n2, mem_n2)
            spk_inh, mem_inh = self.lif_inh(cur_inh, mem_inh)

            # --- Warstwa 2: N3 (zbiera N1 + N2, hamowany przez N_inh) ---
            cur_n3 = (spk_n1 * w_n3_1 + spk_n2 * w_n3_2 + spk_inh * w_inh_n3)
            spk_n3, mem_n3 = self.lif_n3(cur_n3, mem_n3)

            spk_n1_rec.append(spk_n1)
            spk_n2_rec.append(spk_n2)
            spk_n3_rec.append(spk_n3)
            spk_inh_rec.append(spk_inh)

        # Stack spike trains: (timesteps, batch) → (batch, timesteps)
        spk_n1_all = torch.stack(spk_n1_rec, dim=0).permute(1, 0)
        spk_n2_all = torch.stack(spk_n2_rec, dim=0).permute(1, 0)
        spk_n3_all = torch.stack(spk_n3_rec, dim=0).permute(1, 0)
        spk_inh_all = torch.stack(spk_inh_rec, dim=0).permute(1, 0)

        # Trigger = spike rate N3 (fraction of timesteps with spikes)
        trigger = spk_n3_all.mean(dim=1, keepdim=True)  # (batch, 1)

        neuron_spikes = {
            "N1": spk_n1_all,
            "N2": spk_n2_all,
            "N3": spk_n3_all,
            "N_inh": spk_inh_all,
        }

        return trigger, neuron_spikes

    def get_weights_dict(self) -> Dict[str, float]:
        """Zwraca słownik aktualnych wag (do logowania/eksportu).

        Returns:
            Słownik {nazwa_wagi: wartość_float}.
        """
        return {
            "w_n1": self.w_n1.item(),
            "w_n2": self.w_n2.item(),
            "w_n3_from_n1": self.w_n3_from_n1.item(),
            "w_n3_from_n2": self.w_n3_from_n2.item(),
            "w_inh": self.w_inh.item(),
            "w_inh_to_n3": self.w_inh_to_n3.item(),
        }

    def get_thresholds_dict(self) -> Dict[str, float]:
        """Zwraca słownik aktualnych progów V_th.

        Returns:
            Słownik {neuron: V_th_float}.
        """
        return {
            "vth_n1": self.vth_n1.item(),
            "vth_n2": self.vth_n2.item(),
            "vth_n3": self.vth_n3.item(),
            "vth_inh": self.vth_inh.item(),
        }

    def set_quantize_mode(self, mode: str) -> None:
        """Ustawia tryb kwantyzacji.

        Args:
            mode: "none", "hat", "gumbel", "qat".
        """
        assert mode in ("none", "hat", "gumbel", "qat"), f"Nieznany tryb: {mode}"
        self.quantize_mode = mode

    def enable_mismatch(self, enabled: bool = True) -> None:
        """Włącza/wyłącza symulację mismatch hardware.

        Args:
            enabled: True = dodaj szum, False = bez szumu.
        """
        self.mismatch_enabled = enabled

    def clamp_weights(self) -> None:
        """Clampuje wagi do fizycznie sensownego zakresu.

        FIX v2: Inhibitor chroniony — w_inh min 0.15, w_inh_to_n3 max -0.20.
        Zapobiega degenerate collapse gdzie gradient zabija hamowanie.
        """
        with torch.no_grad():
            self.w_n1.clamp_(0.05, 0.95)
            self.w_n2.clamp_(0.05, 0.95)
            self.w_n3_from_n1.clamp_(0.05, 0.95)
            self.w_n3_from_n2.clamp_(0.05, 0.95)
            # FIX: Inhibitor chroniony — minimum 0.15 (było 0.05)
            self.w_inh.clamp_(0.15, 0.95)
            # FIX: Hamowanie chronione — max -0.20 (było -0.05)
            self.w_inh_to_n3.clamp_(-0.95, -0.20)
            # Progi V_th
            self.vth_n1.clamp_(0.1, 0.95)
            self.vth_n2.clamp_(0.1, 0.95)
            self.vth_n3.clamp_(0.1, 0.95)
            self.vth_inh.clamp_(0.1, 0.95)
            # Scale factors (QAT)
            self.scale_n1.clamp_(HW_CONFIG.r_syn_min, HW_CONFIG.r_syn_max)
            self.scale_n2.clamp_(HW_CONFIG.r_syn_min, HW_CONFIG.r_syn_max)
            self.scale_n3_1.clamp_(HW_CONFIG.r_syn_min, HW_CONFIG.r_syn_max)
            self.scale_n3_2.clamp_(HW_CONFIG.r_syn_min, HW_CONFIG.r_syn_max)
            self.scale_inh.clamp_(HW_CONFIG.r_syn_min, HW_CONFIG.r_syn_max)
