#!/usr/bin/env python3
"""
run_search.py — sweep po liczbie neuronów: osobny run GA dla każdego N.

Dla każdego N z --neurons uruchamia niezależny GA (stałe N, ewoluuje okablowanie
i układ warstw), zbiera najlepszy genom i F1, na końcu wypisuje tabelę
"F1 vs liczba neuronów" i zapisuje wyniki do JSON + CSV. Opcjonalnie
(--train-winner) dotrenuje zwycięzcę pełnym cyklem HAT->QAT i wyeksportuje
jego nastawy płytek (hw_config.json + tabela kalibracyjna).

Przykłady:
    python run_search.py --neurons 4 6 8 10 --mode synth --out wyniki_synth
    python run_search.py --neurons 4 6 8 10 --mode real \
        --data ../architecture_14_neurons_patryk_09_07/spikes_manifest7/train \
        --val-data ../architecture_14_neurons_patryk_09_07/spikes_manifest7/val \
        --arch-dir ../architecture_14_neurons_patryk_09_07 \
        --epochs 4 --pop 24 --gens 15 --train-winner --winner-epochs 60 --out wyniki_real
"""
from __future__ import annotations

import argparse
import json
import time

from ga import GAConfig, run_ga


def main():
    ap = argparse.ArgumentParser(description="Sweep GA po liczbie neuronów SNN Lu.i")
    ap.add_argument("--neurons", type=int, nargs="+", default=[4, 6, 8, 10],
                    help="lista N (neurony ukryte + decyzyjny) do porównania")
    ap.add_argument("--mode", choices=["synth", "real"], default="synth")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--max-hidden-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ga_results", help="prefiks plików wyjściowych")
    # opcje trybu real:
    ap.add_argument("--data", default=None, help="katalog spike-CSV (tryb real)")
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--arch-dir", default="../architecture_14_neurons_patryk_09_07")
    ap.add_argument("--limit", type=int, default=None,
                    help="max plikow/klase (proxy). UWAGA: wartosc != cache zmusza "
                         "SpikeClips do przebudowy cache w katalogu danych; jesli "
                         "katalog jest read-only (WinError 5), zostaw puste, by uzyc "
                         "gotowego _cache_*.npz")
    ap.add_argument("--epochs", type=int, default=4, help="epoki proxy-treningu w GA")
    ap.add_argument("--num-samples", type=int, default=6000)
    # dotrenowanie zwyciezcy:
    ap.add_argument("--train-winner", action="store_true",
                    help="po sweepie dotrenuj najlepsza topologie pelnym HAT->QAT "
                         "i wyeksportuj hw_config.json + tabele nastaw (tylko real)")
    ap.add_argument("--winner-epochs", type=int, default=60,
                    help="epoki pelnego treningu zwyciezcy")
    ap.add_argument("--winner-per-n", action="store_true",
                    help="dotrenuj i wyeksportuj zwyciezce KAZDEGO N, nie tylko globalnego")
    args = ap.parse_args()

    # zbuduj funkcję fitness
    if args.mode == "synth":
        from fitness import synth_fitness
        fitness = synth_fitness
        rf = None
        print("[tryb] SYNTH - heurystyka bez treningu (tylko test mechaniki GA)")
    else:
        if not args.data:
            ap.error("--mode real wymaga --data (katalog spike-CSV)")
        from fitness import RealFitness
        rf = RealFitness(arch_dir=args.arch_dir, data=args.data,
                         val_data=args.val_data, limit=args.limit,
                         epochs=args.epochs, num_samples=args.num_samples,
                         seed=args.seed)
        fitness = rf
        print(f"[tryb] REAL - proxy-trening ({args.epochs} epok, limit {args.limit})")

    results = []
    for n in args.neurons:
        t0 = time.time()
        cfg = GAConfig(n_total=n, pop_size=args.pop, generations=args.gens,
                       elite=args.elite, max_hidden_layers=args.max_hidden_layers,
                       seed=args.seed)
        res = run_ga(fitness, cfg, log=print)
        dt = time.time() - t0
        best = res.best
        results.append({
            "n_total": n,
            "fitness": best.fitness,
            "topology": best.genome.to_dict(),
            "history": res.history,
            "evaluated": res.evaluated,
            "seconds": round(dt, 1),
        })
        print(f"\n=== N={n}: best fitness={best.fitness:.4f} "
              f"({res.evaluated} ocen, {dt:.0f}s) ===")
        print(best.genome.pretty())
        print()

    # zapis
    js = f"{args.out}.json"
    with open(js, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    csv = f"{args.out}.csv"
    with open(csv, "w", encoding="utf-8") as f:
        f.write("n_total,fitness,hidden_layers,layer_sizes,evaluated,seconds\n")
        for r in results:
            t = r["topology"]
            sizes = "-".join(str(x) for x in t["layer_sizes"])
            f.write(f"{r['n_total']},{r['fitness']:.4f},{t['hidden_layers']},"
                    f"{sizes},{r['evaluated']},{r['seconds']}\n")

    # tabela podsumowująca
    print("\n" + "=" * 52)
    print(f"{'N':>4} | {'fitness':>8} | {'warstwy':>18} | ocen")
    print("-" * 52)
    for r in sorted(results, key=lambda x: x["fitness"], reverse=True):
        sizes = "-".join(str(x) for x in r["topology"]["layer_sizes"])
        print(f"{r['n_total']:>4} | {r['fitness']:>8.4f} | {sizes:>18} | {r['evaluated']}")
    print("=" * 52)
    print(f"\nzapisano: {js} , {csv}")

    # dotrenowanie i eksport nastaw zwyciezcy
    if args.train_winner:
        if args.mode != "real":
            print("\n[train-winner] pominieto: wymaga --mode real (potrzebne dane).")
            return
        from genome import Genome
        from winner import export_genome_config, train_full

        winners = results if args.winner_per_n else \
            [max(results, key=lambda x: x["fitness"])]
        for w in winners:
            n = w["n_total"]
            print(f"\n########## PELNY TRENING ZWYCIEZCY N={n} "
                  f"(sweep-fitness {w['fitness']:.4f}) ##########")
            g = Genome.from_dict(w["topology"])
            print(g.pretty())
            model, m = train_full(rf, g, epochs=args.winner_epochs,
                                  ckpt=f"{args.out}_winner_N{n}.pt")
            cfg_path = f"{args.out}_hw_config_N{n}.json"
            export_genome_config(model, cfg_path,
                                 extra={"winner_val_metrics": m,
                                        "n_total": n,
                                        "sweep_fitness": w["fitness"]})
            print(f"[train-winner] N={n}: F1={m.get('f1', 0):.3f} -> {cfg_path}")


if __name__ == "__main__":
    main()
