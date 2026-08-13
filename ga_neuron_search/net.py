#!/usr/bin/env python3
"""
net.py — buduje sieć SNN z genomu, wykorzystując LuiLayer z snn_hw_pipeline.

GenomeNet uogólnia LuiNet (który miał zaszyte 3 warstwy) na dowolny warstwowy
genom. Reużywa DOKŁADNIE tej samej dynamiki płytki Lu.i (LuiLayer), stałych
sprzętowych i modelu neuronu — GA zmienia tylko topologię, nie fizykę.

Wymaga, by folder architektury był na sys.path (patrz fitness.py / run_search.py).
"""
from __future__ import annotations

import sys
from typing import List

import torch
import torch.nn as nn

# import z głównego pipeline'u (te same stałe i warstwa co w treningu HW)
from snn_hw_pipeline import CH_IN, LuiLayer, V_TH  # noqa: E402

from genome import Genome


class GenomeNet(nn.Module):
    """Sieć Lu.i o topologii zadanej genomem.

    forward zwraca:
        hidden : lista spike-tensorów warstw ukrytych [B,T,n_k]
        so     : spiki neuronu decyzyjnego [B,T,1]
        vo     : membrana neuronu decyzyjnego [B,T]  (do BCE po vmax jak w oryginale)
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

    # zgodność z LuiNet
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


def genome_loss(out, y, pos_weight, rate_lo=0.02, rate_hi=0.30, margin_w=0.5):
    """Uogólniona strata (odpowiednik loss_fn z pipeline'u dla dowolnej liczby warstw).

    Decyzja po vmax membrany wyjściowej (7. dioda Lu.i) + margines na tle +
    regularyzacja aktywności wszystkich warstw ukrytych (ani martwe, ani zasycone).
    """
    import torch.nn.functional as F
    vmax = out["vo"].max(dim=1).values
    logit = 6.0 * (vmax - V_TH)
    bce = F.binary_cross_entropy_with_logits(
        logit, y, pos_weight=torch.tensor(pos_weight, device=y.device))

    neg = (y < 0.5).float()
    margin = ((vmax - 0.80).clamp(min=0) ** 2 * neg).sum() / neg.sum().clamp(min=1)

    reg = 0.0
    for sh in out["hidden"]:
        r = sh.mean(dim=(0, 1))
        reg = reg + ((rate_lo - r).clamp(min=0) ** 2).sum() \
                  + ((r - rate_hi).clamp(min=0) ** 2).sum()

    return bce + margin_w * margin + 0.2 * reg, logit


@torch.no_grad()
def genome_eval_f1(model, loader, dev):
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        vmax = model(x)["vo"].max(dim=1).values
        p = (vmax >= V_TH).float()
        tp += ((p == 1) & (y == 1)).sum().item()
        fp += ((p == 1) & (y == 0)).sum().item()
        fn += ((p == 0) & (y == 1)).sum().item()
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    f1 = 2 * pre * rec / max(pre + rec, 1e-9)
    return {"recall": rec, "precision": pre, "f1": f1}
