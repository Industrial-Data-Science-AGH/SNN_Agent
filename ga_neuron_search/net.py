#!/usr/bin/env python3
"""
net.py — buduje sieć SNN z genomu, wykorzystując LuiLayer z snn_hw_pipeline.

GenomeNet uogólnia LuiNet (zaszyte 3 warstwy) na dowolny warstwowy genom.
Reużywa tej samej dynamiki płytki Lu.i (LuiLayer), stałych i modelu neuronu —
GA zmienia tylko topologię, nie fizykę.

Ocena (#1): NIE zwijamy okna do vmax. Decyzja i metryki idą po LICZBIE SPIKÓW
neuronu decyzyjnego D w czasie ("k spików w oknie") i są agregowane do poziomu
KLIPU — tak jak działa demo na sprzęcie (evaluate_events w pipeline).
"""
from __future__ import annotations

import sys
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from snn_hw_pipeline import CH_IN, LuiLayer, V_TH  # noqa: E402

from genome import Genome


class GenomeNet(nn.Module):
    """Sieć Lu.i o topologii zadanej genomem.

    forward zwraca:
        hidden : lista spike-tensorów warstw ukrytych [B,T,n_k]
        so     : spiki neuronu decyzyjnego [B,T,1]
        vo     : membrana neuronu decyzyjnego [B,T]
    """

    def __init__(self, genome: Genome, hw=None, quantize: bool = False):
        super().__init__()
        assert genome.is_valid(), f"niepoprawny genom: {genome.violations()}"
        self.genome = genome
        self.layers_mod = nn.ModuleList()
        prev = CH_IN
        for k, layer in enumerate(genome.layers):
            n_post = len(layer)
            names = [f"L{k+1}n{i}" for i in range(n_post)]
            self.layers_mod.append(
                LuiLayer(prev, n_post, layer, names, hw=hw, quantize=quantize))
            prev = n_post

    def forward(self, x):
        hidden: List[torch.Tensor] = []
        s = x
        for i, layer in enumerate(self.layers_mod):
            s, v = layer(s)
            if i < len(self.layers_mod) - 1:
                hidden.append(s)
        return {"hidden": hidden, "so": s, "vo": v.squeeze(-1)}

    def layers(self):
        return list(self.layers_mod)

    def set_quantize(self, flag: bool):
        for l in self.layers_mod:
            l.quantize = flag

    def set_mismatch(self, flag: bool):
        for l in self.layers_mod:
            l.mismatch_active = flag

    def freeze_signs(self):
        for l in self.layers_mod:
            l.freeze_signs()


# ============================================================ strata (czasowa)

def genome_loss(out, y, pos_weight, rate_lo=0.02, rate_hi=0.30, margin_w=0.5,
                spk_w=0.5, k_ref=2.0):
    """Strata z członem CZASOWYM.

    Składniki:
      * BCE po vmax membrany D — daje gęsty gradient od pierwszej epoki (neuron
        rzadko strzela na starcie, więc sam człon spikowy by nie ruszył),
      * człon spikowy (temporal): D ma dać >= k_ref spików na pozytywach i 0 na
        tle — różniczkowalny przez surrogate-gradient na so. To on egzekwuje
        dekoder "k spików w oknie", którego brakowało,
      * margines na tle (membrana tła nie podchodzi pod próg),
      * regularyzacja aktywności warstw ukrytych (ani martwe, ani zasycone).
    """
    vmax = out["vo"].max(dim=1).values
    logit = 6.0 * (vmax - V_TH)
    bce = F.binary_cross_entropy_with_logits(
        logit, y, pos_weight=torch.tensor(pos_weight, device=y.device))

    pos = (y > 0.5).float()
    neg = 1.0 - pos
    margin = ((vmax - 0.80).clamp(min=0) ** 2 * neg).sum() / neg.sum().clamp(min=1)

    nspk = out["so"].sum(dim=(1, 2)).clamp(max=5.0)  # liczba spików D w oknie [B]
    spk = ((k_ref - nspk).clamp(min=0) * pos).sum() / pos.sum().clamp(min=1) \
        + (nspk * neg).sum() / neg.sum().clamp(min=1)

    reg = 0.0
    for sh in out["hidden"]:
        r = sh.mean(dim=(0, 1))
        reg = reg + ((rate_lo - r).clamp(min=0) ** 2).sum() \
                  + ((r - rate_hi).clamp(min=0) ** 2).sum()

    total = bce + margin_w * margin + spk_w * spk + 0.2 * reg
    return total, logit


# ============================================================ ocena zdarzeniowa

def _average_precision(y_true, score) -> float:
    """AP (pole pod krzywą precision-recall) — bezprogowa, odporna na niebalans."""
    y_true = np.asarray(y_true).astype(float)
    order = np.argsort(-np.asarray(score))
    y = y_true[order]
    P = y.sum()
    if P == 0:
        return 0.0
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / P
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


@torch.no_grad()
def genome_eval_events(model, win, lab, fidx, dev, k=2, bs=256):
    """Metryki na poziomie KLIPU (jak na żywym demie).

    Klip alarmuje, gdy KTÓREŚ jego okno ma >= k spików neuronu D. Zwraca
    clip-F1/recall/precision/fa oraz AP po ciągłym score (max spików D w klipie).
    Argumenty win/lab/fidx to tablice numpy (okna, etykiety, indeks pliku-klipu).
    """
    model.eval()
    n = len(lab)
    nspk = np.zeros(n, dtype=np.float32)
    for lo in range(0, n, bs):
        x = torch.from_numpy(win[lo:lo + bs]).float().to(dev)
        nspk[lo:lo + x.shape[0]] = model(x)["so"].sum(dim=(1, 2)).cpu().numpy()

    files = np.unique(fidx)
    f_lab = np.array([lab[fidx == f].max() for f in files])
    f_spk = np.array([nspk[fidx == f].max() for f in files])

    det = f_spk >= k
    tp = float(((det == 1) & (f_lab == 1)).sum())
    fp = float(((det == 1) & (f_lab == 0)).sum())
    fn = float(((det == 0) & (f_lab == 1)).sum())
    rec = tp / max(tp + fn, 1.0)
    pre = tp / max(tp + fp, 1.0)
    f1 = 2 * pre * rec / max(pre + rec, 1e-9)
    fa = fp / max(float((f_lab == 0).sum()), 1.0)
    ap = _average_precision(f_lab, f_spk)
    return {"clip_f1": f1, "clip_recall": rec, "clip_precision": pre,
            "clip_fa_rate": fa, "ap": ap, "k": int(k),
            "n_pos": int((f_lab == 1).sum()), "n_neg": int((f_lab == 0).sum())}
