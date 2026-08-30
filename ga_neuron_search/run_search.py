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
import time

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
                    help="max plikow/klase; wartosc != cache zmusza przebudowe "
                         "(WinError 5 w read-only) — zostaw puste, by uzyc _cache_*.npz")
    ap.add_argument("--epochs", type=int, default=4, help="epoki proxy-treningu w GA")
    ap.add_argument("--num-samples", type=int, default=6000)
    ap.add_argument("--k", type=int, default=1, help="dekoder: >=k spikow D = alarm (k=1 spojnie)")
    ap.add_argument("--metric", choices=["ap", "clip_f1", "recall_fa"], default="clip_f1",
                    help="metryka fitness: clip_f1 | ap | recall_fa (recall @ budzet FA/h, "
                         "decyzyjna metryka z DATASET_CONTRACT)")
    ap.add_argument("--stream-budget", type=float, default=6.0,
                    help="budzet FA/h dla metric=recall_fa (domyslnie 6/h)")
    ap.add_argument("--fitness-seeds", type=int, default=1,
                    help="usrednij fitness po tylu seedach (mniejsza wariancja)")
    ap.add_argument("--feature-penalty", type=float, default=0.005,
                    help="kara za liczbe uzytych kanalow encodera (selekcja cech; 0 = wylaczona)")
    ap.add_argument("--channels-head", type=int, default=None,
                    help="uzyj tylko pierwszych N kanalow danych (A/B: np. 7 = same HW w spikes_ext)")
    ap.add_argument("--quiet", action="store_true", help="bez logu per-kandydat")
    # dotrenowanie zwyciezcy:
    ap.add_argument("--train-winner", action="store_true")
    ap.add_argument("--winner-epochs", type=int, default=60)
    ap.add_argument("--winner-per-n", action="store_true")
    args = ap.parse_args()

    if args.mode == "synth":
        from fitness import synth_fitness
        fitness, rf = synth_fitness, None
        print("[tryb] SYNTH - heurystyka bez treningu")
    else:
        if not args.data:
            ap.error("--mode real wymaga --data")
        from fitness import RealFitness
        rf = RealFitness(arch_dir=args.arch_dir, data=args.data,
                         val_data=args.val_data, test_data=args.test_data,
                         limit=args.limit,
                         epochs=args.epochs, num_samples=args.num_samples,
                         k=args.k, metric=args.metric, fitness_seeds=args.fitness_seeds,
                         feature_penalty=args.feature_penalty,
                         channels_head=args.channels_head,
                         stream_budget=args.stream_budget,
                         verbose=not args.quiet, seed=args.seed)
        fitness = rf
        # pula kanałów encodera = to, co jest w danych (nie zaszyte 7) — GA wybiera
        from genome import configure_features
        configure_features(rf.n_channels, rf.channel_names)
        print(f"[tryb] REAL - proxy-trening ({args.epochs} epok, k={args.k}, "
              f"metryka={args.metric}, seedy={args.fitness_seeds}, "
              f"kanaly={rf.n_channels}: {','.join(rf.channel_names)})")

    results = []
    for n in args.neurons:
        t0 = time.time()
        cfg = GAConfig(n_total=n, pop_size=args.pop, generations=args.gens,
                       elite=args.elite, max_hidden_layers=args.max_hidden_layers,
                       screen_mult=args.screen_mult, screen_budget=args.screen_budget,
                       seed=args.seed)
        res = run_ga(fitness, cfg, log=print)
        dt = time.time() - t0
        feats = res.best.genome.feature_names_used()
        results.append({
            "n_total": n, "fitness": res.best.fitness,
            "topology": res.best.genome.to_dict(), "history": res.history,
            "features_used": feats, "n_features_used": len(feats),
            "evaluated": res.evaluated, "seconds": round(dt, 1),
        })
        print(f"\n=== N={n}: best {args.metric}={res.best.fitness:.4f} "
              f"({res.evaluated} ocen, {dt:.0f}s) ===")
        print(res.best.genome.pretty())
        print(f"  kanaly encodera ({len(feats)}): {', '.join(feats)}")
        print()

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
        from winner import export_genome_config, train_full
        winners = results if args.winner_per_n else [chosen]
        for w in winners:
            n = w["n_total"]
            print(f"\n########## PELNY TRENING ZWYCIEZCY N={n} "
                  f"(sweep-F1 {w['fitness']:.4f}) ##########")
            g = Genome.from_dict(w["topology"])
            print(g.pretty())
            model, m = train_full(rf, g, epochs=args.winner_epochs,
                                  ckpt=f"{args.out}_winner_N{n}.pt")
            cfg_path = f"{args.out}_hw_config_N{n}.json"
            export_genome_config(model, cfg_path, channels=rf.channel_names,
                                 extra={"winner_val_metrics": m, "n_total": n,
                                        "sweep_fitness": w["fitness"]})
            print(f"[train-winner] N={n}: clip-F1={m.get('clip_f1', 0):.3f} -> {cfg_path}")
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
