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
                    help="Liczba procesów do równoległej oceny (domyślnie liczba rdzeni CPU). "
                        "Uwaga: Każdy worker ładuje własną kopię cache'u danych w RAM "
                        "(architektura N+1 kopii w pamięci głównej). Na maszynach z małą ilością "
                        "RAM-u ustawienie dużej wartości (np. 18) może spowodować błąd OOM.")
    ap.add_argument("--device", default=None,
                    help="cuda | cpu; domyślnie cuda-jak-dostępne, inaczej cpu "
                         "(na Macu CPU — MPS zmierzone ~3x wolniejsze)")
    # successive-halving (#4):
    ap.add_argument("--screen-mult", type=int, default=1,
                    help="oceń screen_mult*pop losowych osobników tanim budżetem, "
                         "zatrzymaj najlepsze pop (1 = wyłączone)")
    ap.add_argument("--screen-budget", type=float, default=0.34,
                    help="ułamek epok na screening")
    ap.add_argument("--parsimony-eps", type=float, default=0.0,
                    help="wybierz najmniejsze N w zasięgu eps clip-F1 od najlepszego")
    # tryb real:
    ap.add_argument("--data", default=None)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--test-data", default=None,
                    help="nietkniety split testowy — raport koncowy zwyciezcy (recall @ FA/h)")
    ap.add_argument("--arch-dir", default="../architecture_14_neurons_patryk_09_07")
    ap.add_argument("--limit", type=int, default=None,
                    help="max plikow/klase; wartość != cache zmusza przebudowe "
                         "(WinError 5 w read-only) — zostaw puste, by uzyc _cache_*.npz")
    ap.add_argument("--epochs", type=int, default=4, help="epoki proxy-treningu w GA")
    ap.add_argument("--num-samples", type=int, default=6000)
    ap.add_argument("--k", type=int, default=2, help="dekoder: >= k spikow D = alarm")
    ap.add_argument("--metric", choices=["ap", "clip_f1", "recall_fa"], default="clip_f1",
                    help="metryka fitness: clip_f1 | ap | recall_fa (recall @ budzet FA/h, "
                         "decyzyjna metryka z DATASET_CONTRACT)")
    ap.add_argument("--stream-budget", type=float, default=6.0,
                    help="budzet FA/h dla metric=recall_fa (domyslnie 6/h)")
    ap.add_argument("--fitness-seeds", type=int, default=1,
                    help="usrednij fitness po tylu seedach (mniejsza wariancja)")
    ap.add_argument("--pos-weight", type=float, default=1.0,
                        help="waga klasy pozytywnej w BCE proxy-treningu i dotrenowania")
    ap.add_argument("--feature-penalty", type=float, default=0.005,
                    help="kara za liczbe uzytych kanalow encodera (selekcja cech; 0 = wylaczona)")
    ap.add_argument("--channels-head", type=int, default=None,
                    help="uzyj tylko pierwszych N kanalow danych (A/B: np. 7 = same HW w spikes_ext)")
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
                         val_data=args.val_data, test_data=args.test_data, limit=args.limit,
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
        f.write("n_total,fitness,hidden_layers,layer_sizes,n_features_used,features_used,evaluated,seconds\n")
        for r in results:
            t = r["topology"]
            sizes = "-".join(str(x) for x in t["layer_sizes"])
            feats = "|".join(r.get("features_used", []))
            f.write(f"{r['n_total']},{r['fitness']:.4f},{t['hidden_layers']},"
                    f"{sizes},{r.get('n_features_used', 0)},{feats},"
                    f"{r['evaluated']},{r['seconds']}\n")

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
    print(f"[parsymonia] najlepszy {args.metric}={best_fit:.4f}; w zasięgu eps={args.parsimony_eps} "
          f"-> wybór N={chosen['n_total']} ({args.metric}={chosen['fitness']:.4f}, "
          f"kanały={','.join(chosen.get('features_used', []))})")
    print(f"zapisano: {js} , {csv}")

    # === RAPORT ZWYCIEZCY NA NIETKNIETYM TEST (fix trzeciego splitu) ===
    # Selekcja szla na VAL; nagłówkowy wynik liczymy na TEST, którego nikt nie
    # dotykał. Bez --train-winner robimy szybki proxy-trening zwyciezcy.
    test_ready = rf is not None and args.test_data
    if test_ready and not args.train_winner:
        from genome import Genome
        from stream_eval import format_report, primary_recall, report_to_dict
        print(f"\n### ZWYCIEZCA N={chosen['n_total']} NA TEST "
              f"(nietkniety split, proxy-trening) ###")
        rep = rf.evaluate_on_test(Genome.from_dict(chosen["topology"]), epochs=args.epochs)
        print(format_report(rep))
        chosen["test_recall_at_budget"] = round(primary_recall(rep, args.stream_budget), 4)
        chosen["test_budget_fa_h"] = args.stream_budget
        chosen["test_stream"] = report_to_dict(rep)   # recall @1 i @6 FA/h, per kind, CI
        with open(js, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[TEST] recall @ {args.stream_budget:g} FA/h = "
              f"{chosen['test_recall_at_budget']:.3f}  <- nagłówkowy wynik (nietkniety split)")

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
            best_model, best_m_val, best_m_test, best_pw = None, None, None, None
            for pw in grid:
                print(f"\n----- pos_weight={pw} (HAT->QAT, {args.winner_epochs} ep) -----")
                ckpt = (f"{args.out}_winner_N{n}.pt" if len(grid) == 1
                        else f"{args.out}_winner_N{n}_pw{pw}.pt")
                model, m_val, m_test = train_full(rf, g, epochs=args.winner_epochs,
                                                  pos_weight=pw, ckpt=ckpt)
                print(f"[pos-weight] pw={pw}: val clipF1 {m_val.get('clip_f1', 0):.3f} "
                      f"val FA {m_val.get('clip_fa_rate', 0):.3f} -> {ckpt}")
                if best_m_val is None or m_val.get("clip_f1", 0) > best_m_val.get("clip_f1", 0):
                    best_model, best_m_val, best_m_test, best_pw = model, m_val, m_test, pw
            model, m_val, m_test, pw = best_model, best_m_val, best_m_test, best_pw

            extra = {
                "winner_val_metrics": m_val,
                "winner_test_metrics": m_test,
                "n_total": n,
                "sweep_fitness": w["fitness"],
                "pos_weight": pw
            }
            if len(grid) > 1:
                tuned_ckpt = f"{args.out}_tuned_winner_N{n}.pt"
                torch.save({"model": model.state_dict(), "val_metrics": m_val,
                            "test_metrics": m_test, "topology": g.to_dict(),
                            "pos_weight": pw}, tuned_ckpt)
                cfg_path = f"{args.out}_tuned_hw_config_N{n}.json"
                print(f"[pos-weight] wybor pw={pw} (val clipF1 {m_val.get('clip_f1', 0):.3f}) "
                      f"-> {tuned_ckpt}")
            else:
                cfg_path = f"{args.out}_hw_config_N{n}.json"

            if args.tune_k:
                best_k, km, table = tune_k(model, rf, k_range=args.tune_k)
                extra["tuned_k"] = best_k
                extra["tune_k_table"] = table
                print(f"[tune-k] N={n}: wybor k={best_k} (val clipF1 {km['clip_f1']:.3f}, "
                      f"FA {km['clip_fa_rate']:.3f})")

            export_genome_config(model, cfg_path, channels=rf.channel_names, extra=extra)
            print(f"[train-winner] N={n}: val clip-F1={m_val.get('clip_f1', 0):.3f} -> {cfg_path}")
            # nagłówkowy wynik na NIETKNIETYM tescie (z pełnego modelu, nie proxy)
            if test_ready:
                from stream_eval import format_report, primary_recall, report_to_dict
                rep = rf.evaluate_on_test(g, model=model)
                print(f"### N={n} NA TEST (nietkniety split, pełny trening) ###")
                print(format_report(rep))
                w["test_recall_at_budget"] = round(primary_recall(rep, args.stream_budget), 4)
                w["test_budget_fa_h"] = args.stream_budget
                w["test_stream"] = report_to_dict(rep)
        with open(js, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
