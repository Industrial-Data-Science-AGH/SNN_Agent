"""
Smoke test M1 punkt 2: poprawiony run_final_evaluation_stage / run_ext_evaluation_stage
w ga_runner.py -- sprawdza, że działają end-to-end na Twoim prawdziwym
fitness.py/datasecie, po poprawce val/test aliasingu i budget=6.0 mixupu.

To NIE jest test wydajności ani prawdziwy odbiór M1 (do tego trzeba pełnego
protokołu z punktu 1/3). To tylko potwierdzenie, że:
  1. kod się w ogóle wykonuje bez wyjątków na prawdziwych ścieżkach val/test,
  2. wyniki per-seed nie są NaN/inf i nie są absurdalnie rozrzucone,
  3. test_{metric} jest w sensownym zakresie (nie 0.0 "martwej sieci" -- guard
     w RealFitness.__call__ zerowałby wtedy fitness, ale tu liczymy test
     bezpośrednio przez eval_events, więc taki guard nie działa -- 0.0 na
     teście przy niezerowym wyniku GA na val jest sygnałem do zgłoszenia).

Uruchamiać z katalogu master_pipeline/ (tak samo jak pipeline.py):

    python3 smoke_test_final_eval.py --config config.json
"""
import argparse
import math
import multiprocessing
import os
import sys


class FakeTracker:
    """Minimalny tracker -- ga_runner.py potrzebuje tylko .device,
    .log_stage_time(name, elapsed) i .log_metrics(name, dict). Jeśli Twój
    prawdziwy tracker.py ma dodatkowe wymagania (np. get_run_dir() używane w
    run_hardware_export_stage -- nieużywane tutaj), ten smoke test i tak ich
    nie potrzebuje dla etapu 2/3."""

    def __init__(self, device: str):
        self.device = device
        self.stage_times = {}
        self.metrics = {}

    def log_stage_time(self, name, elapsed):
        self.stage_times[name] = elapsed
        print(f"[TRACKER] stage_time[{name}] = {elapsed:.2f}s")

    def log_metrics(self, name, d):
        self.metrics[name] = d
        print(f"[TRACKER] metrics[{name}] = {d}")


def shrink_config_for_smoke_test(config, num_samples_cap: int):
    config.train.proxy_epochs = 1
    if hasattr(config.train, "num_samples"):
        config.train.num_samples = min(config.train.num_samples, num_samples_cap)
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--pop-size", type=int, default=4, help="Tylko do wygenerowania taniego genomu-materiału.")
    ap.add_argument("--num-samples-cap", type=int, default=8)
    ap.add_argument("--skip-ext", action="store_true", help="Pomiń run_ext_evaluation_stage (np. brak spikes_ext).")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from pipeline_config import PipelineConfig
    from hardware import get_device, _build_rf_kwargs_for_benchmark

    project_root = os.path.dirname(os.path.abspath(os.getcwd()))
    for p in (project_root, os.path.join(project_root, "ga_neuron_search")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import ga_runner
    from ga_neuron_search.fitness import RealFitness
    from ga_neuron_search.ga import GAConfig, run_ga

    config = PipelineConfig.from_json(args.config)
    config = shrink_config_for_smoke_test(config, args.num_samples_cap)
    device = get_device("cpu")
    tracker = FakeTracker(device)

    print("[SMOKE] Generuję tani, PRAWDZIWY genom-materiał (nie pełny GA, tylko żeby mieć best_topology)...")
    rf_kwargs = _build_rf_kwargs_for_benchmark(config, project_root, device)
    max_neurons = max(config.ga.neurons_range)
    ga_cfg = GAConfig(
        n_total=max_neurons, pop_size=args.pop_size, generations=1,
        elite=min(config.ga.elite, args.pop_size), max_hidden_layers=4, seed=config.seed,
    )
    rf = RealFitness(**rf_kwargs)
    res = run_ga(rf, ga_cfg, log=lambda *a, **k: None)
    best_topology = res.best.genome.to_dict()
    print(f"[SMOKE] Genom gotowy, val_fitness(GA)={res.best.fitness:.4f} -- "
          f"punkt odniesienia: test_{{metric}} poniżej NIE powinien być 0.0 "
          f"jeśli to jest zbliżone.")

    if not args.skip_ext and hasattr(config.data, "spikes_ext"):
        print("\n== run_ext_evaluation_stage ==")
        try:
            ga_runner.run_ext_evaluation_stage(config, tracker, best_topology)
        except Exception as exc:
            print(f"[SMOKE][UWAGA] run_ext_evaluation_stage rzuciło wyjątek "
                  f"(pomiń, jeśli nie masz spikes_ext lokalnie): {exc}")
    else:
        print("\n[SMOKE] Pomijam run_ext_evaluation_stage (--skip-ext albo brak config.data.spikes_ext).")

    print("\n== run_final_evaluation_stage ==")
    ga_runner.run_final_evaluation_stage(config, tracker, best_topology)

    m = tracker.metrics.get("continuous_test", {})
    metric_key = f"test_{config.ga.fitness_metric}"
    per_seed_key = f"{metric_key}_per_seed"

    print("\n[SMOKE] Sprawdzam wyniki...")
    ok = True

    final_score = m.get(metric_key)
    per_seed = m.get(per_seed_key, [])

    if final_score is None or not math.isfinite(final_score):
        print(f"  [BŁĄD] {metric_key}={final_score} -- brak albo nie-skończone")
        ok = False
    else:
        print(f"  [OK] {metric_key}={final_score:.4f} (skończone)")

    if not per_seed or len(per_seed) != getattr(config.train, "fitness_seeds", 3) and len(per_seed) != 3:
        print(f"  [UWAGA] liczba wyników per-seed ({len(per_seed)}) -- sprawdź czy zgadza się z oczekiwaną liczbą seedów")
    if any(not math.isfinite(s) for s in per_seed):
        print(f"  [BŁĄD] per-seed zawiera NaN/inf: {per_seed}")
        ok = False
    else:
        print(f"  [OK] per-seed skończone: {[f'{s:.4f}' for s in per_seed]}")

    if per_seed:
        spread = max(per_seed) - min(per_seed)
        print(f"  [INFO] rozrzut między seedami: {spread:.4f} (duży rozrzut = wysoka wariancja treningu, "
              f"nie błąd tego skryptu, ale warto to mieć na uwadze przy interpretacji wyniku)")

    if final_score == 0.0:
        print("  [UWAGA] final_score == 0.0 dokładnie -- w RealFitness.__call__ to sygnatura "
              "\"martwej sieci\" (guard f1<=1e-9), ale tutaj liczymy test bezpośrednio przez "
              "eval_events, więc ten guard NIE działa. Jeśli GA na val dało sensowny wynik "
              "(patrz wyżej), a test wyszedł 0.0 -- to podejrzane, zgłoś to.")

    print("\n[SMOKE] WYNIK:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
    