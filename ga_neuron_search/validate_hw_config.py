#!/usr/bin/env python3
"""
validate_hw_config.py — round-trip walidacja eksportu nastaw płytek.

Odtwarza sieć GENOMOWĄ z `hw_config_*.json` (trymer %, znak +/-, okablowanie,
tau_syn/tau_mem, V_leak -> p_tsyn/p_tmem/p_vleak) i porównuje metryki na zbiorze
walidacyjnym z tymi zapisanymi w checkpoincie (`winner_val_metrics`).

Łapie te same błędy, które zepsułyby wdrożenie na płytki:
  * odwrócone znaki wag,
  * zgubione synapsy (wpis w configu, ale brak połączenia w masce),
  * trymery poniżej rozdzielczości (pot_pct < --min-pot -> waga zerowana).

To jest programowa namiastka walidacji na sprzęcie: realne porównanie to
`snn_hw_pipeline.py compare` z nagraniami z hardware (nie istnieją jeszcze dla
tej konfiguracji). `calibrate.py` NIE jest walidacją płytek — kalibruje budżet
epok proxy-treningu.

Użycie:
    ../.venv/bin/python validate_hw_config.py \
        --config wyniki_real_hw_config_N8.json \
        --arch-dir ../architecture_14_neurons_patryk_09_07 \
        --data  ../architecture_14_neurons_patryk_09_07/spikes_manifest7/train \
        --val-data ../architecture_14_neurons_patryk_09_07/spikes_manifest7/val \
        --k 2 --tol 0.02
"""
from __future__ import annotations

import argparse
import json

import torch

from genome import Genome


def rebuild_from_config(cfg: dict, min_pot: float = 5.0):
    """Zbuduj GenomeNet z cfg['boards'] (odwrotność export_genome_config).

    Zwraca (model, list[warningi]). Wagi poniżej rozdzielczości trymera
    (pot_pct < min_pot) są ZEROWANE — tyle da się ustawić na płytce.
    """
    import net
    from snn_hw_pipeline import (CHANNELS, TAU_MEM_RANGE, TAU_SYN_RANGE,
                                 V_LEAK_MAX, V_TH, W_MAX, _inv_map)

    g = Genome.from_dict(cfg["topology"])
    model = net.GenomeNet(g, hw=None, quantize=True)

    layers = model.layers()
    pre_names = [list(CHANNELS)] + [l.names for l in layers[:-1]]
    warnings, zeroed = [], []

    with torch.no_grad():
        for li, (layer, pres) in enumerate(zip(layers, pre_names)):
            col_of = {pre: j for j, pre in enumerate(pres)}
            for i, name in enumerate(layer.names):
                board = cfg["boards"].get(name)
                if board is None:
                    warnings.append(f"[!] brak wpisu w konfigu dla {name} — neuron pominięty")
                    continue
                k_scale = board["scale_k"]
                for syn in board["synapses"]:
                    col = col_of.get(syn["from"])
                    if col is None or int(layer.mask[i, col].item()) == 0:
                        warnings.append(
                            f"[!] {name}.{syn['port']} ({syn['from']}) — brak "
                            f"połączenia w masce topologii (config niezgodny)")
                        continue
                    pot = syn["pot_pct"]
                    if pot < min_pot:
                        zeroed.append((name, syn["from"], pot))
                        layer.w[i, col] = 0.0
                    else:
                        sign = 1.0 if syn["sign"] == "+" else -1.0
                        layer.w[i, col] = sign * (pot / 100.0) * W_MAX / k_scale
                layer.p_tsyn[i] = _inv_map(board["tau_syn_ms"] / 1000.0, *TAU_SYN_RANGE)
                layer.p_tmem[i] = _inv_map(board["tau_mem_ms"] / 1000.0, *TAU_MEM_RANGE)
                vl_sim = V_TH - (V_TH - board["v_leak"]) / k_scale
                layer.p_vleak[i] = _inv_map(vl_sim, 0.0, V_LEAK_MAX)

    return model, warnings, zeroed


def main():
    ap = argparse.ArgumentParser(description="Round-trip walidacja hw_config (software)")
    ap.add_argument("--config", required=True, help="wyeksportowany hw_config_*.json")
    ap.add_argument("--arch-dir", default="../architecture_14_neurons_patryk_09_07")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--k", type=int, default=None, help="próg dekodera (domyślnie z configu)")
    ap.add_argument("--tol", type=float, default=0.02, help="tolerancja |Δ clip-F1|")
    ap.add_argument("--min-pot", type=float, default=5.0,
                    help="pot_pct poniżej tej wartości -> waga zerowana")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = json.load(f)

    # te same dane walidacyjne co w treningu (ten sam val_cap subsample) — nie
    # zależą od epochs/metric/num_samples, więc wystarczy domyślny RealFitness.
    from fitness import RealFitness
    rf = RealFitness(arch_dir=args.arch_dir, data=args.data, val_data=args.val_data,
                     verbose=False)
    if rf.device != "cpu":
        print(f"[val] device={rf.device} — dla determinizmu round-tripu użyję cpu")
        rf.device = "cpu"

    import net
    model, warnings, zeroed = rebuild_from_config(cfg, min_pot=args.min_pot)

    k = args.k or cfg.get("winner_val_metrics", {}).get("k", 2)
    m = net.genome_eval_events(model, rf.va_win, rf.va_lab, rf.va_fidx, rf.device, k=k)

    recorded = cfg.get("winner_val_metrics") or {}
    rec_f1 = recorded.get("clip_f1", None)
    print(f"\n[round-trip] k={k}  clip-F1 {m['clip_f1']:.4f}  "
          f"rec {m['clip_recall']:.3f}  prec {m['clip_precision']:.3f}  "
          f"FA {m['clip_fa_rate']:.3f}  AP {m['ap']:.3f}")
    print(f"[roundtrip] zapisane w config: clip-F1 {rec_f1 if rec_f1 is not None else '?'}")

    for w in warnings:
        print(w)
    for name, src, pot in zeroed:
        print(f"[i] {name} <- {src}: pot {pot:.1f}% < {args.min_pot}% — synapsa wyzerowana "
              f"(poniżej rozdzielczości trymera)")

    # tabela okablowania
    print("\nplytka  wejscia (port=zrodlo znak trymer%)")
    for n, b in cfg["boards"].items():
        s = "  ".join(f"{x['port']}={x['from']}{x['sign']}{x['pot_pct']:.0f}%"
                      for x in b["synapses"])
        print(f"{n:7}  {s}")

    if rec_f1 is not None:
        d = abs(m["clip_f1"] - rec_f1)
        ok = d <= args.tol
        print(f"\n[roundtrip] Δclip-F1 = {d:.4f} (tol {args.tol})"
              f" -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    print("\n[roundtrip] brak winner_val_metrics w configu — tylko rekonstrukcja")


if __name__ == "__main__":
    raise SystemExit(main())
