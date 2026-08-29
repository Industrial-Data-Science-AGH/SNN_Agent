#!/usr/bin/env python3
"""
export_winner.py — pełny trening JEDNEJ topologii i eksport nastaw płytek.

Most między "znalezieniem topologii" (sweep GA, tu) a "treningiem gdzie indziej"
(np. na GPU). Bierze zwycięską topologię z wyniku sweepu (`wyniki_*.json`),
robi pełny cykl HAT->QAT NA TEJ MASZYNIE, na której odpalisz, i zapisuje
`hw_config.json` + checkpoint. Nie powtarza całego sweepu.

Wybór topologii: --n <N> (konkretne N z JSON-a) albo domyślnie najlepszy fitness.

Użycie (najlepiej na maszynie z GPU):
    python export_winner.py --winner-json wyniki_demo.json --n 8 --epochs 60 \
        --arch-dir ../architecture_14_neurons_patryk_09_07 \
        --data ../architecture_14_neurons_patryk_09_07/spikes_manifest7/train \
        --val-data ../architecture_14_neurons_patryk_09_07/spikes_manifest7/val \
        --out winner8
"""
from __future__ import annotations

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner-json", required=True, help="wynik sweepu (wyniki_*.json)")
    ap.add_argument("--n", type=int, default=None,
                    help="które N wziąć; domyślnie topologia o najlepszym fitness")
    ap.add_argument("--epochs", type=int, default=60, help="epoki pełnego HAT->QAT")
    ap.add_argument("--k", type=int, default=1, help="dekoder: >=k spikow D = alarm (k=1 spojnie)")
    ap.add_argument("--arch-dir", default="../architecture_14_neurons_patryk_09_07")
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--out", default="winner", help="prefiks plików wyjściowych")
    args = ap.parse_args()

    results = json.load(open(args.winner_json, encoding="utf-8"))
    if args.n is not None:
        cand = [r for r in results if r["n_total"] == args.n]
        if not cand:
            raise SystemExit(f"brak N={args.n} w {args.winner_json} "
                             f"(dostępne: {[r['n_total'] for r in results]})")
        r = cand[0]
    else:
        r = max(results, key=lambda x: x["fitness"])

    from fitness import RealFitness
    from genome import Genome
    from winner import export_genome_config, train_full

    g = Genome.from_dict(r["topology"])
    print(f"[winner] N={r['n_total']}  sweep-fitness={r['fitness']:.4f}")
    print(g.pretty())

    rf = RealFitness(arch_dir=args.arch_dir, data=args.data, val_data=args.val_data,
                     epochs=12, k=args.k, verbose=False)
    # pula kanałów encodera = to, co jest w danych (spójne z siecią z sweepu)
    from genome import configure_features
    configure_features(rf.n_channels, rf.channel_names)
    model, m = train_full(rf, g, epochs=args.epochs, ckpt=f"{args.out}_winner.pt")
    cfg_path = f"{args.out}_hw_config.json"
    export_genome_config(model, cfg_path, channels=rf.channel_names,
                         extra={"winner_val_metrics": m, "n_total": r["n_total"],
                                "sweep_fitness": r["fitness"],
                                "source_json": args.winner_json})
    print(f"\n[gotowe] clip-F1={m.get('clip_f1', 0):.3f} AP={m.get('ap', 0):.3f} "
          f"-> {cfg_path}  (+ {args.out}_winner.pt)")


if __name__ == "__main__":
    main()
