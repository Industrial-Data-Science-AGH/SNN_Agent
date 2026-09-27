"""
Smoke test M0 krok 4: checkpoint/resume eksperymentu GA na PRAWDZIWYM
RealFitness (Twój dataset), nie na fitness syntetycznym.

To dopełnienie test_resume.py (który dowodzi poprawności samej mechaniki
checkpoint/resume w ga.py na deterministycznym fitness syntetycznym, bez
zależności od datasetu). Tutaj sprawdzamy to samo, ale na rzeczywistym
pipeline: czy RealFitness jest wystarczająco deterministyczny (dla ustalonego
genomu i config), żeby przerwany+wznowiony bieg GA dał TEN SAM wynik co bieg
bez przerwy -- to jest właśnie kryterium odbioru M0 ("Test wznowienia
odtwarza stan eksperymentu").

WAŻNE: to jest smoke test poprawności (małe, tanie ustawienia), nie benchmark
wydajności. Uruchamiać z katalogu master_pipeline/ (jak main.py):

    python3 smoke_test_ga_resume.py --config config.json

Wymaga załatanego ga.py z checkpoint_path (ten sam plik co wysłałem) w
ga_neuron_search/.
"""
import argparse
import multiprocessing
import os
import sys


def shrink_config_for_smoke_test(config, num_samples_cap: int):
    """Ten sam wzorzec co w smoke_test_benchmark.py -- małe proxy_epochs/num_samples
    tylko w pamięci tego procesu, żeby test był tani."""
    config.train.proxy_epochs = 1
    if hasattr(config.train, "num_samples"):
        config.train.num_samples = min(config.train.num_samples, num_samples_cap)
    return config


class CrashAfterN:
    """Owija RealFitness i rzuca RuntimeError po dokładnie N wywołaniach --
    symuluje padnięcie procesu W ŚRODKU generacji (nie na jej granicy)."""

    def __init__(self, fn, crash_after: int):
        self.fn = fn
        self.crash_after = crash_after
        self.calls = 0

    def __call__(self, g, *args):
        self.calls += 1
        if self.calls > self.crash_after:
            raise RuntimeError(f"[SYMULOWANA AWARIA] po {self.crash_after} wywołaniach fitness")
        return self.fn(g, *args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--pop-size", type=int, default=4)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--crash-after", type=int, default=None,
                     help="Domyślnie: pop_size + 2 (padnij w 1. generacji, po paru dzieciach).")
    ap.add_argument("--num-samples-cap", type=int, default=8)
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from pipeline_config import PipelineConfig
    from hardware import get_device, _build_rf_kwargs_for_benchmark

    project_root = os.path.dirname(os.path.abspath(os.getcwd()))
    for p in (project_root, os.path.join(project_root, "ga_neuron_search")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from ga_neuron_search.fitness import RealFitness
    from ga_neuron_search.ga import GAConfig, run_ga, load_checkpoint_summary

    config = PipelineConfig.from_json(args.config)
    config = shrink_config_for_smoke_test(config, args.num_samples_cap)
    device = get_device("cpu")
    max_neurons = max(config.ga.neurons_range)

    crash_after = args.crash_after if args.crash_after is not None else args.pop_size + 2
    ckpt_path = os.path.join(os.getcwd(), "_smoke_ga_resume_checkpoint.json")
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    def fresh_rf():
        rf_kwargs = _build_rf_kwargs_for_benchmark(config, project_root, device)
        return RealFitness(**rf_kwargs)

    ga_cfg = GAConfig(
        n_total=max_neurons, pop_size=args.pop_size, generations=args.generations,
        elite=min(config.ga.elite, args.pop_size), max_hidden_layers=4,
        seed=config.seed, patience=100,  # patience wysokie: chcemy pełne `generations`, bez early stop
    )

    print(f"[SMOKE] pop_size={args.pop_size}, generations={args.generations}, "
          f"crash_after={crash_after} wywołań fitness, num_samples="
          f"{getattr(config.train, 'num_samples', '<brak>')}, proxy_epochs={config.train.proxy_epochs}")

    print("\n== Baseline (bez przerwy) ==")
    baseline = run_ga(fresh_rf(), ga_cfg, log=print)
    print(f"[SMOKE] baseline: best={baseline.best.fitness:.6f} evaluated={baseline.evaluated}")

    print("\n== Przerwanie w środku GA (symulowana awaria) ==")
    crashy = CrashAfterN(fresh_rf(), crash_after=crash_after)
    try:
        run_ga(crashy, ga_cfg, log=print, checkpoint_path=ckpt_path)
        print("[SMOKE] UWAGA: oczekiwano awarii, ale GA zakończyło się normalnie -- "
              "podnieś --crash-after albo --generations.")
    except RuntimeError as exc:
        print(f"[SMOKE] Awaria zasymulowana zgodnie z oczekiwaniem: {exc}")

    if not os.path.exists(ckpt_path):
        print("[SMOKE][BŁĄD] Brak checkpointu po awarii -- prawdopodobnie padło jeszcze "
              "w trakcie oceny populacji startowej (przed pierwszym punktem zapisu). "
              "Podnieś --crash-after.")
        sys.exit(1)

    summary = load_checkpoint_summary(ckpt_path)
    print(f"[SMOKE] Checkpoint po awarii: gen={summary['gen']}, "
          f"best={summary['best_fitness']:.6f}, evaluated={summary['evaluated']}")

    print("\n== Wznowienie z checkpointu ==")
    resumed = run_ga(fresh_rf(), ga_cfg, log=print, checkpoint_path=ckpt_path)
    print(f"[SMOKE] resumed: best={resumed.best.fitness:.6f} evaluated={resumed.evaluated}")

    print("\n[SMOKE] Sprawdź:")
    ok = True
    if baseline.best.fitness != resumed.best.fitness:
        print(f"  [BŁĄD] best_fitness różni się: {baseline.best.fitness} != {resumed.best.fitness}")
        ok = False
    else:
        print(f"  [OK] best_fitness identyczny: {baseline.best.fitness:.6f}")
    if baseline.evaluated != resumed.evaluated:
        print(f"  [BŁĄD] evaluated różni się: {baseline.evaluated} != {resumed.evaluated}")
        ok = False
    else:
        print(f"  [OK] evaluated identyczny: {baseline.evaluated}")
    if baseline.history != resumed.history:
        print(f"  [BŁĄD] history różni się:\n    baseline={baseline.history}\n    resumed ={resumed.history}")
        ok = False
    else:
        print(f"  [OK] history identyczna ({len(baseline.history)} generacji)")
    if baseline.best.genome.layers != resumed.best.genome.layers:
        print("  [BŁĄD] topologia najlepszego genomu różni się między baseline a wznowionym biegiem")
        ok = False
    else:
        print("  [OK] topologia najlepszego genomu identyczna")
    if os.path.exists(ckpt_path):
        print("  [BŁĄD] checkpoint powinien zniknąć po ukończonym eksperymencie, a wciąż istnieje")
        ok = False
        os.remove(ckpt_path)
    else:
        print("  [OK] checkpoint usunięty po ukończeniu")

    print("\n[SMOKE] WYNIK:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
    