#!/usr/bin/env python3
"""
run_search.py — sweep po liczbie neuronów: osobny run GA dla każdego N.

Dla każdego N osobny GA (stałe N, ewoluuje okablowanie i warstwy). Fitness to
clip-F1 z oceny ZDARZENIOWEJ (dekoder k-spików D). Na końcu tabela "F1 vs N",
zapis JSON/CSV, wybór zwycięzcy z PARSYMONIĄ (najmniejsze N w zasięgu eps od
najlepszego) i opcjonalny pełny trening + eksport nastaw (--train-winner).

Przykład:
    python run_search.py --neurons 4 6 8 10 --mode real \
        --data ../architecture_14_neurons_patryk_09_07/spikes_manifest7/train \
        --val-data ../architecture_14_neurons_patryk_09_07/spikes_manifest7/val \
        --arch-dir ../architecture_14_neurons_patryk_09_07 \
        --epochs 10 --pop 30 --gens 20 --screen-mult 3 --parsimony-eps 0.02 \
        --train-winner --winner-epochs 60 --out wyniki_real
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from ga import GAConfig, run_ga


def main():
    ap = argparse.ArgumentParser(description="Sweep GA po liczbie neuronów SNN Lu.i")
    ap.add_argument("--neurons", type=int, nargs="+", default=[4, 6, 8, 10])
    ap.add_argument("--mode", choices=["synth", "real"], default="synth")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=15)
    ap.add_argument("--elite", type=int, default=3)
    ap.add_argument("--max-hidden-layers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ga_results")
    ap.add_argument("--workers", type=int, default=None,
                    help="liczba procesów ewaluacji (real, CPU); None = os.cpu_count(), "
                         "1 = sekwencyjnie (jak dotąd)")
    ap.add_argument("--device", default=None,
                    help="cuda | cpu; domyślnie cuda-jak-dostępne, inaczej cpu "
                         "(na Macu CPU — MPS zmierzone ~3x wolniejsze)")
    # successive-halving (#4):
    ap.add_argument("--screen-mult", type=int, default=3,
                    help="oceń screen_mult*pop losowych osobników tanim budżetem, "
                         "zatrzymaj najlepsze pop (1 = wyłączone)")
    ap.add_argument("--screen-budget", type=float, default=0.34,
                    help="ułamek epok na screening")
    ap.add_argument("--parsimony-eps", type=float, default=0.0,
                    help="wybierz najmniejsze N w zasięgu eps clip-F1 od najlepszego")
    # tryb real:
    ap.add_argument("--data", default=None)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--arch-dir", default="../architecture_14_neurons_patryk_09_07")
    ap.add_argument("--limit", type=int, default=None,
                    help="max plikow/klase; wartość != cache zmusza przebudowe "
                         "(WinError 5 w read-only) — zostaw puste, by uzyc _cache_*.npz")
    ap.add_argument("--epochs", type=int, default=4, help="epoki proxy-treningu w GA")
    ap.add_argument("--num-samples", type=int, default=6000)
    ap.add_argument("--k", type=int, default=2, help="dekoder: >= k spikow D = alarm")
    ap.add_argument("--metric", choices=["ap", "clip_f1"], default="ap",
                    help="metryka fitness: ap (bezprogowa, mniej szumna) lub clip_f1")
    ap.add_argument("--fitness-seeds", type=int, default=3,
                    help="usrednij fitness po tylu seedach (mniejsza wariancja)")
    ap.add_argument("--pos-weight", type=float, default=3.0,
                    help="waga klasy pozytywnej w BCE proxy-treningu i dotrenowania")
    ap.add_argument("--quiet", action="store_true", help="bez logu per-kandydat")
    # dotrenowanie zwyciezcy:
    ap.add_argument("--train-winner", action="store_true")
    ap.add_argument("--winner-epochs", type=int, default=60)
    ap.add_argument("--winner-per-n", action="store_true")
    ap.add_argument("--pos-weight-grid", type=float, nargs="+", default=None,
                    help="dotrenuj zwycięzcę dla każdej pos_weight i zapisz najlepszą "
                         "(np. --pos-weight-grid 1.5 2.0 3.0)")
    ap.add_argument("--tune-k", type=int, nargs="+", default=None,
                    help="przebieg progu dekodera k na wytrenowanym zwycięzcy "
                         "(np. --tune-k 1 2 3 4 5 6)")
    args = ap.parse_args()

    # determinizm + brak thrasha: małe SNN nie korzystają z wielu wątków matmul,
    # a jeden wątek w procesie głównym == jeden wątek w workerach puli (porównywalne
    # wyniki sekwencyjne i równoległe).
    if args.mode == "real":
        torch.set_num_threads(1)

    if args.mode == "synth":
        from fitness import synth_fitness
        fitness, rf = synth_fitness, None
        print("[tryb] SYNTH - heurystyka bez treningu")
    else:
        if not args.data:
            ap.error("--mode real wymaga --data")
        from fitness import ParallelFitness, RealFitness
        rf_kwargs = dict(arch_dir=args.arch_dir, data=args.data,
                         val_data=args.val_data, limit=args.limit,
                         epochs=args.epochs, num_samples=args.num_samples,
                         k=args.k, metric=args.metric, fitness_seeds=args.fitness_seeds,
                         pos_weight=args.pos_weight,
                         verbose=not args.quiet, seed=args.seed, device=args.device)
        rf = RealFitness(**rf_kwargs)
        if args.workers == 1:
            fitness = rf
        else:
            fitness = ParallelFitness(rf_kwargs, max_workers=args.workers)
        print(f"[tryb] REAL - proxy-trening ({args.epochs} epok, k={args.k}, "
              f"metryka={args.metric}, seedy={args.fitness_seeds}, "
              f"workers={getattr(fitness, 'max_workers', 1)}, pos_weight={args.pos_weight})")

    results = []
    try:
        for n in args.neurons:
            t0 = time.time()
            cfg = GAConfig(n_total=n, pop_size=args.pop, generations=args.gens,
                           elite=args.elite, max_hidden_layers=args.max_hidden_layers,
                           screen_mult=args.screen_mult, screen_budget=args.screen_budget,
                           seed=args.seed)
            res = run_ga(fitness, cfg, log=print)
            dt = time.time() - t0
            results.append({
                "n_total": n, "fitness": res.best.fitness,
                "topology": res.best.genome.to_dict(), "history": res.history,
                "evaluated": res.evaluated, "seconds": round(dt, 1),
            })
            print(f"\n=== N={n}:  best clip-F1={res.best.fitness:.4f} "
                  f"({res.evaluated} ocen, {dt:.0f}s) ===")
            print(res.best.genome.pretty())
            print()
    finally:
        if hasattr(fitness, "close"):
            fitness.close()

    js, csv = f"{args.out}.json", f"{args.out}.csv"
    with open(js, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    with open(csv, "w", encoding="utf-8") as f:
        f.write("n_total,fitness,hidden_layers,layer_sizes,evaluated,seconds\n")
        for r in results:
            t = r["topology"]
            sizes = "-".join(str(x) for x in t["layer_sizes"])
            f.write(f"{r['n_total']},{r['fitness']:.4f},{t['hidden_layers']},"
                    f"{sizes},{r['evaluated']},{r['seconds']}\n")

    print("\n" + "=" * 52)
    print(f"{'N':>4} | {args.metric:>8} | {'warstwy':>18} | ocen")
    print("-" * 52)
    for r in sorted(results, key=lambda x: x["fitness"], reverse=True):
        sizes = "-".join(str(x) for x in r["topology"]["layer_sizes"])
        print(f"{r['n_total']:>4} | {r['fitness']:>8.4f} | {sizes:>18} | {r['evaluated']}")
    print("=" * 52)

    # parsymonia: najmniejsze N w zasięgu eps od najlepszego clip-F1
    best_fit = max(r["fitness"] for r in results)
    cand = [r for r in results if r["fitness"] >= best_fit - args.parsimony_eps]
    chosen = min(cand, key=lambda r: r["n_total"])
    print(f"[parsymonia] najlepszy F1={best_fit:.4f}; w zasięgu eps={args.parsimony_eps} "
          f"-> wybór N={chosen['n_total']} (F1={chosen['fitness']:.4f})")
    print(f"zapisano: {js} , {csv}")

    if args.train_winner:
        if args.mode != "real":
            print("\n[train-winner] pominieto: wymaga --mode real.")
            return
        from genome import Genome
        from winner import export_genome_config, train_full, tune_k
        winners = results if args.winner_per_n else [chosen]
        for w in winners:
            n = w["n_total"]
            print(f"\n########## PELNY TRENING ZWYCIEZCY N={n} "
                  f"(sweep-F1 {w['fitness']:.4f}) ##########")
            g = Genome.from_dict(w["topology"])
            print(g.pretty())
            grid = args.pos_weight_grid or [args.pos_weight]
            best_model, best_m, best_pw = None, None, None
            for pw in grid:
                print(f"\n----- pos_weight={pw} (HAT->QAT, {args.winner_epochs} ep) -----")
                ckpt = (f"{args.out}_winner_N{n}.pt" if len(grid) == 1
                        else f"{args.out}_winner_N{n}_pw{pw}.pt")
                model, m = train_full(rf, g, epochs=args.winner_epochs,
                                      pos_weight=pw, ckpt=ckpt)
                print(f"[pos-weight] pw={pw}: clipF1 {m.get('clip_f1', 0):.3f} "
                      f"FA {m.get('clip_fa_rate', 0):.3f} -> {ckpt}")
                if best_m is None or m.get("clip_f1", 0) > best_m.get("clip_f1", 0):
                    best_model, best_m, best_pw = model, m, pw
            model, m, pw = best_model, best_m, best_pw

            extra = {"winner_val_metrics": m, "n_total": n,
                     "sweep_fitness": w["fitness"], "pos_weight": pw}
            if len(grid) > 1:
                tuned_ckpt = f"{args.out}_tuned_winner_N{n}.pt"
                torch.save({"model": model.state_dict(), "metrics": m,
                            "topology": g.to_dict(), "pos_weight": pw}, tuned_ckpt)
                cfg_path = f"{args.out}_tuned_hw_config_N{n}.json"
                print(f"[pos-weight] wybor pw={pw} (clipF1 {m.get('clip_f1', 0):.3f}) "
                      f"-> {tuned_ckpt}")
            else:
                cfg_path = f"{args.out}_hw_config_N{n}.json"

            if args.tune_k:
                best_k, km, table = tune_k(model, rf, k_range=args.tune_k)
                extra["tuned_k"] = best_k
                extra["tune_k_table"] = table
                print(f"[tune-k] N={n}: wybor k={best_k} (clipF1 {km['clip_f1']:.3f}, "
                      f"FA {km['clip_fa_rate']:.3f})")

            export_genome_config(model, cfg_path, extra=extra)
            print(f"[train-winner] N={n}: clip-F1={m.get('clip_f1', 0):.3f} -> {cfg_path}")


if __name__ == "__main__":
    main()
