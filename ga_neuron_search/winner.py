#!/usr/bin/env python3
"""
winner.py — pełny trening zwycięskiej topologii i eksport nastaw płytek Lu.i.

  * train_full(rf, genome, ...) — pełny harmonogram HAT->QAT (jak
    snn_hw_pipeline.train), checkpoint po ocenie ZDARZENIOWEJ (clip-F1).
  * export_genome_config(model, path, ...) — uogólnienie export_config na dowolną
    liczbę warstw: per płytka trymer% + znak +/-, pasek LED (V_leak),
    tau_syn/tau_mem, pulses_to_fire do weryfikacji sim<->hw.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import torch
import torch.nn as nn

import net
from genome import FEATURE_NAMES, Genome
from snn_hw_pipeline import DT, V_TH, W_DEADZONE, W_MAX, CHANNELS, pulses_to_fire


# ============================================================ pełny trening


def train_full(
    rf,
    genome: Genome,
    epochs: int = 60,
    hat_frac: float = 0.4,
    lr: float = 3e-3,
    pos_weight: float = 3.0,
    patience: int = 20,
    ckpt: Optional[str] = None,
    log=print,
):
    """Pełny cykl HAT->QAT dla zadanej topologii. rf = instancja RealFitness
    (dane + urządzenie + ocena zdarzeniowa)."""
    torch_, np = rf.torch, rf.np
    dev = rf.device
    torch_.manual_seed(0)
    np.random.seed(0)

    model = net.GenomeNet(genome, hw=None, quantize=False).to(dev)
    opt = torch_.optim.Adam(model.parameters(), lr=lr)
    sched = torch_.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    dl_tr = rf.DataLoader(
        rf.tr_ds, batch_size=128, sampler=rf.make_sampler(rf.tr_lab, 12000)
    )

    hat_epochs = int(round(hat_frac * epochs))
    freeze_ep = int(0.8 * epochs)
    log(
        f"[winner] plan: HAT 0..{hat_epochs - 1}, QAT {hat_epochs}..{epochs - 1}, "
        f"znaki zamrozone od {freeze_ep}"
    )

    best, since, best_state, best_m = -1.0, 0, None, {}
    for ep in range(epochs):
        t0 = time.time()
        phase = "HAT" if ep < hat_epochs else "QAT"
        if ep == hat_epochs:
            model.set_quantize(True)
            best, since = -1.0, 0
            log(f"[winner ep {ep}] start QAT: kwantyzacja on, reset best")
        if ep == freeze_ep:
            model.freeze_signs()
            log(f"[winner ep {ep}] znaki wag zamrozone")

        model.train()
        model.set_mismatch(True)
        loss_sum, nb = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            loss, _ = net.genome_loss(model(x), y, pos_weight, k_ref=rf.k)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            nb += 1
        model.set_mismatch(False)
        sched.step()

        m = rf.eval_events(model)
        tag = ""
        if m["clip_f1"] > best:
            best, since, tag = m["clip_f1"], 0, " *"
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_m = m
        else:
            since += 1
        log(
            f"[winner ep {ep:3d} {phase}] loss {loss_sum / max(nb, 1):.4f} "
            f"clipF1 {m['clip_f1']:.3f} AP {m['ap']:.3f} rec {m['clip_recall']:.3f} "
            f"FA {m['clip_fa_rate']:.3f} {time.time() - t0:.0f}s{tag}"
        )
        if since >= patience and phase == "QAT":
            log(f"[winner] early stop (QAT, brak poprawy {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    if ckpt:
        torch.save(
            {
                "model": model.state_dict(),
                "metrics": best_m,
                "topology": genome.to_dict(),
            },
            ckpt,
        )
    log(
        f"[winner] najlepszy clip-F1={best_m.get('clip_f1', 0):.3f} "
        f"(AP {best_m.get('ap', 0):.3f})"
    )
    return model, best_m


# ============================================================ eksport nastaw


def export_genome_config(model, path: str, channels=None, extra: Optional[dict] = None):
    """Uogólniony export_config: dowolna liczba warstw GenomeNet -> hw_config.json."""
    channels = channels or CHANNELS
    layers = model.layers()
    pre_names = [channels] + [l.names for l in layers[:-1]]

    cfg = {
        "dt_s": DT,
        "v_th": V_TH,
        "channels": list(channels),
        "topology": model.genome.to_dict(),
        "boards": {},
    }
    if extra:
        cfg.update(extra)

    V_LEAK_MIN_HW = 0.20 * V_TH
    for layer, pres in zip(layers, pre_names):
        W = layer.weights().detach()
        vl, ts, tm = (
            layer.v_leak().detach(),
            layer.tau_syn().detach(),
            layer.tau_mem().detach(),
        )
        for i, name in enumerate(layer.names):
            w = W[i]
            m = w.abs().max().item()
            if m < W_DEADZONE:
                print(
                    f"[!] {name}: wszystkie wagi w martwej strefie — neuron nieuzywany"
                )
                continue
            k_allow = (V_TH - V_LEAK_MIN_HW) / max(V_TH - vl[i].item(), 1e-3)
            k = min(W_MAX / m, k_allow)
            v_leak_hw = V_TH - k * (V_TH - vl[i].item())

            syn = []
            for j, pre in enumerate(pres):
                if layer.mask[i, j] == 0:
                    continue
                wij = w[j].item()
                pot = 100.0 * abs(wij) * k / W_MAX
                if pot < 5.0:
                    print(
                        f"[i] {name}.J{len(syn) + 1} ({pre}): pot {pot:.1f}% — "
                        f"ponizej rozdzielczosci trymera"
                    )
                syn.append(
                    {
                        "port": f"J{len(syn) + 1}",
                        "from": pre,
                        "sign": "+" if wij >= 0 else "-",
                        "pot_pct": round(pot, 1),
                        "w_sim": round(wij, 4),
                        "pulses_to_fire_100Hz": pulses_to_fire(
                            abs(wij) * k, ts[i].item(), tm[i].item(), v_leak_hw
                        ),
                    }
                )
            cfg["boards"][name] = {
                "tau_syn_ms": round(1000 * ts[i].item(), 1),
                "tau_mem_ms": round(1000 * tm[i].item(), 1),
                "v_leak": round(v_leak_hw, 3),
                "led_bar_pct": round(50.0 * v_leak_hw / V_TH, 1),
                "scale_k": round(k, 3),
                "synapses": syn,
            }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\n[export] {path}")
    print(
        f"{'plytka':8} {'tau_syn':>8} {'tau_mem':>9} {'pasek LED':>10}  synapsy (wejscie znak trymer%)"
    )
    for n, b in cfg["boards"].items():
        s = "  ".join(
            f"{x['port']}={x['from']}{x['sign']}{x['pot_pct']:.0f}%"
            f"(n*={x['pulses_to_fire_100Hz']})"
            for x in b["synapses"]
        )
        print(
            f"{n:8} {b['tau_syn_ms']:7.1f}ms {b['tau_mem_ms']:8.1f}ms "
            f"{b['led_bar_pct']:9.1f}%  {s}"
        )
    return cfg
