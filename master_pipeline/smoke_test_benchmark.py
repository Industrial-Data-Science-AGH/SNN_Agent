"""
Smoke test M0 (tryb 'real' w benchmark_workers): mały, szybki test poprawności
PRZED uruchomieniem właściwego benchmarku na docelowym sprzęcie (M5 Max).

Nie mierzy wydajności -- sprawdza tylko, czy kod w ogóle działa i czy liczy
to samo niezależnie od liczby workerów. To można zrobić na zwykłym Linuksie,
bo RealFitness/ParallelFitness to zwykły kod CPU, bez zależności od MPS.

WAŻNE: uruchamiaj tym środowiskiem (venv), którym normalnie odpalasz
master_pipeline/main.py -- tym, w którym jest zainstalowane ga_neuron_search
i torch używany do treningu. To prawdopodobnie NIE jest edge_env (ta nazwa
sugeruje środowisko agenta Pi/edge, nie treningu) -- jeśli nie masz pewności,
sprawdź które środowisko ma zainstalowane 'torch' i skąd main.py faktycznie
importuje ga_neuron_search.

Uruchamiać z katalogu master_pipeline/ (tak samo jak main.py), tym samym
poleceniem co zwykłe uruchomienie, tylko innym plikiem:

    python3 smoke_test_benchmark.py --config config.json

Jeśli config.json nie da się bezpiecznie zmniejszyć w pamięci (patrz niżej),
skrypt jawnie o tym poinformuje -- wtedy zrób kopię config.json z małymi
proxy_epochs/num_samples i wskaż ją przez --config, dodając --no-shrink.
"""
import argparse
import multiprocessing

from pipeline_config import PipelineConfig
from hardware import get_device, benchmark_workers


def shrink_config_for_smoke_test(config, num_samples_cap: int):
    """
    Zmniejsza koszt jednego przebiegu GA, nie dotykając config.json na dysku.

    Uwaga: w konfigu Marcela `config.train` NIE ma pola `num_samples` w ogóle
    (jest tylko proxy_epochs, winner_epochs, hat_frac, lr, batch_size,
    pos_weight_grid, tune_k_range, fitness_seeds) -- mimo że run_ga_stage w
    ga_runner.py czyta `config.train.num_samples`. Jedno z dwojga:
    (a) PipelineConfig ma dla tego pola wartość domyślną -- wtedy działa,
    (b) go nie ma -- wtedy run_ga_stage i tak by się wysypał na AttributeError,
    z tym samym błędem co produkcyjny trening, nie tylko ten smoke test.
    Diagnozujemy to jawnie zamiast zgadywać "niemutowalny".
    """
    print(
        f"[SMOKE] config.train przed zmianą: "
        f"proxy_epochs={getattr(config.train, 'proxy_epochs', '<brak atrybutu>')}, "
        f"num_samples={getattr(config.train, 'num_samples', '<brak atrybutu -- patrz uwaga niżej>')}"
    )

    try:
        original_epochs = config.train.proxy_epochs
        config.train.proxy_epochs = 1
        print(f"[SMOKE] proxy_epochs {original_epochs} -> 1 (tylko w pamięci)")
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(
            f"[BŁĄD] Nie można ustawić config.train.proxy_epochs: {exc}\n"
            "config.train jest prawdopodobnie niemutowalny -- zrób kopię config.json "
            "z małym proxy_epochs i uruchom z --config <kopia> --no-shrink."
        ) from exc

    if not hasattr(config.train, "num_samples"):
        print(
            "[SMOKE] UWAGA: config.train nie ma atrybutu 'num_samples' w Twoim config.json. "
            "Jeśli PipelineConfig nie ma dla niego wartości domyślnej, run_ga_stage "
            "(config.train.num_samples w ga_runner.py) rzuci ten sam AttributeError co "
            "za chwilę ten smoke test -- to NIE jest błąd tego skryptu, tylko istniejąca "
            "luka w config.json/PipelineConfig do sprawdzenia niezależnie od M0."
        )
    else:
        try:
            original_samples = config.train.num_samples
            config.train.num_samples = min(config.train.num_samples, num_samples_cap)
            print(f"[SMOKE] num_samples {original_samples} -> {config.train.num_samples} (tylko w pamięci)")
        except (AttributeError, TypeError) as exc:
            raise RuntimeError(
                f"[BŁĄD] Nie można ustawić config.train.num_samples: {exc}\n"
                "Zrób kopię config.json z małym num_samples i uruchom z "
                "--config <kopia> --no-shrink."
            ) from exc

    return config


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument(
        "--total-tasks", type=int, default=4,
        help="Liczba genomów na konfigurację (mała = szybki test). "
             "Uwaga: config.ga.elite w Twoim config.json to 3 -- hardware.py przytnie "
             "elite do total_tasks jeśli podasz mniej, ale 4 daje margines bez przycinania.",
    )
    ap.add_argument("--worker-counts", default="1,2", help="np. '1,2' lub '1,2,4'.")
    ap.add_argument("--num-samples-cap", type=int, default=8)
    ap.add_argument(
        "--no-shrink",
        action="store_true",
        help="Nie zmniejszaj proxy_epochs/num_samples (użyj, jeśli --config już wskazuje na mały config).",
    )
    args = ap.parse_args()

    worker_counts = tuple(int(x) for x in args.worker_counts.split(","))

    print(f"[SMOKE] Ładowanie configu: {args.config}")
    config = PipelineConfig.from_json(args.config)

    if not args.no_shrink:
        config = shrink_config_for_smoke_test(config, args.num_samples_cap)

    device = get_device(args.device)
    print(f"[SMOKE] Urządzenie: {device}")
    print(
        f"[SMOKE] worker_counts={worker_counts}, total_tasks={args.total_tasks}, "
        f"repeats=1, warmup=False (celowo minimalne -- to tylko test poprawności)"
    )

    best, results = benchmark_workers(
        worker_counts=worker_counts,
        total_tasks=args.total_tasks,
        repeats=1,
        warmup=False,
        config=config,
        device=device,
    )

    print("\n[SMOKE] Zwrócone wartości:", best, results)
    print(
        "\n[SMOKE] Sprawdź w logu powyżej, w tej kolejności:\n"
        "  1. Brak tracebacku przy imporcie ga_neuron_search (RuntimeError na starcie\n"
        "     benchmarku = zły sys.path albo złe środowisko/venv).\n"
        "  2. Dla każdego w: 'evaluated=' równe total_tasks, BEZ linii [UWAGA] o\n"
        "     niezgodności liczby ocenionych genomów.\n"
        "  3. 'best_fitness=' IDENTYCZNY dla wszystkich w, BEZ linii [UWAGA] o różnicy\n"
        "     między konfiguracjami (przy tym samym seedzie różnica = błąd do zgłoszenia).\n"
        "  4. Sam pipeline (ładowanie danych, RealFitness) nie sypie własnymi błędami/\n"
        "     ostrzeżeniami o brakujących plikach czy nieprawidłowym kształcie danych."
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # spójnie z main.py
    main()
    