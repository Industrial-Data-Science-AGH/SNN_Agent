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
import tempfile


class FakeTracker:
    """Minimalny tracker -- ga_runner.py potrzebuje .device, .log_stage_time,
    .log_metrics i (od M2 punkt 2/3, 26.09.2026) .get_run_dir() -- teraz
    używane też w run_final_evaluation_stage (zapis winner_checkpoint.pt) i
    run_hardware_export_stage (katalog hardware_export/), nie tylko w
    hardware exporcie jak wcześniej. Katalog jest tymczasowy per-run smoke
    testu, żeby nie śmiecić w repo."""

    def __init__(self, device: str):
        self.device = device
        self.stage_times = {}
        self.metrics = {}
        self._run_dir = tempfile.mkdtemp(prefix="smoke_run_")

    def log_stage_time(self, name, elapsed):
        self.stage_times[name] = elapsed
        print(f"[TRACKER] stage_time[{name}] = {elapsed:.2f}s")

    def log_metrics(self, name, d):
        self.metrics[name] = d
        print(f"[TRACKER] metrics[{name}] = {d}")

    def get_run_dir(self) -> str:
        return self._run_dir


def shrink_config_for_smoke_test(config, num_samples_cap: int):
    config.train.proxy_epochs = 1
    # M2 punkt 2/3 (26.09.2026): run_final_evaluation_stage trenuje teraz
    # przez winner.train_full (winner_epochs x winner_seeds pelnych,
    # drogich przebiegow HAT->QAT), nie proxy_epochs -- bez przycięcia tych
    # dwoch pol smoke test probowalby odpalic PRODUKCYJNY trening (domyslnie
    # 60 epok x 5 seedow), co nie jest "smoke" tylko pelnym runem.
    if hasattr(config.train, "winner_epochs"):
        config.train.winner_epochs = 2
    if hasattr(config.train, "winner_seeds"):
        config.train.winner_seeds = 1
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
    # M2 punkt 2/3 (26.09.2026): od poprawki nie ma juz `{metric_key}_per_seed`
    # (usredniania kilku odczytow testu) -- jeden odczyt testu dla modelu
    # wybranego jako mediana z winner_seeds przebiegow na val. Ten check byl
    # dopasowany do starego formatu, teraz sprawdzamy zamiast tego obecnosc
    # checkpointu i median_key.
    median_key = f"val_{config.ga.fitness_metric}_median_of_{config.train.winner_seeds}_seeds"

    print("\n[SMOKE] Sprawdzam wyniki (etap 3)...")
    ok = True

    final_score = m.get(metric_key)
    checkpoint_path = m.get("checkpoint_path")
    checkpoint_sha256 = m.get("checkpoint_sha256")
    median_score = m.get(median_key)

    if final_score is None or not math.isfinite(final_score):
        print(f"  [BŁĄD] {metric_key}={final_score} -- brak albo nie-skończone")
        ok = False
    else:
        print(f"  [OK] {metric_key}={final_score:.4f} (skończone, 1 odczyt testu)")

    if median_score is None or not math.isfinite(median_score):
        print(f"  [BŁĄD] {median_key}={median_score} -- brak albo nie-skończone")
        ok = False
    else:
        print(f"  [OK] {median_key}={median_score:.4f}")

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"  [BŁĄD] checkpoint_path={checkpoint_path!r} -- brak pliku na dysku")
        ok = False
    elif not checkpoint_sha256:
        print(f"  [BŁĄD] brak checkpoint_sha256 w metrykach")
        ok = False
    else:
        print(f"  [OK] checkpoint zapisany: {checkpoint_path} (sha256={checkpoint_sha256[:16]}...)")

    if final_score == 0.0:
        print("  [UWAGA] test_{metric} == 0.0 dokładnie -- przy tak małym budżecie smoke testu "
              "(winner_epochs/winner_seeds przycięte) to może być zwykły artefakt niedouczenia, "
              "nie błąd poprawki. Sprawdź na pełnym configu, jeśli to Cię niepokoi.")

    print("\n== run_hardware_export_stage ==")
    try:
        ga_runner.run_hardware_export_stage(config, tracker, best_topology)
        export_info = tracker.metrics.get("hardware_export", {})
        export_path = export_info.get("export_path")
        exported_hash = export_info.get("checkpoint_hash")
        print("\n[SMOKE] Sprawdzam wyniki (etap 4)...")
        if not export_path or not os.path.exists(export_path):
            print(f"  [BŁĄD] export_path={export_path!r} -- brak hw_config.json na dysku")
            ok = False
        else:
            print(f"  [OK] hw_config.json zapisany: {export_path}")
        if exported_hash != checkpoint_sha256:
            print(f"  [BŁĄD] checkpoint_hash w hardware_export ({exported_hash}) != "
                  f"checkpoint_sha256 z continuous_test ({checkpoint_sha256}) -- "
                  f"eksport NIE użył tego samego checkpointu co etap 3 (M2 punkt 2/3 złamane).")
            ok = False
        else:
            print(f"  [OK] eksport użył dokładnie tego samego checkpointu co etap 3 "
                  f"(sha256={exported_hash[:16]}...) -- M2 punkt 2/3: brak ponownego treningu.")
        if not export_info.get("checkpoint_hash_verified_unchanged"):
            print(f"  [BŁĄD] brak potwierdzenia niezmienności wag po eksporcie w metrykach")
            ok = False
        else:
            print(f"  [OK] asercja niezmienności wag w run_hardware_export_stage przeszła "
                  f"(sha256 checkpointu identyczny przed i po eksporcie).")
    except Exception as exc:
        print(f"  [BŁĄD] run_hardware_export_stage rzuciło wyjątek: {exc}")
        ok = False

    print("\n[SMOKE] WYNIK:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
    