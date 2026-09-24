import multiprocessing
import os
import statistics
import sys
import time
import concurrent.futures
import torch


def get_device(device_arg: str = "auto") -> str:
    """
    Wybiera urządzenie obliczeniowe z uwzględnieniem specyfiki Apple Silicon.
    Tryb 'auto' na maszynach z systemem macOS domyślnie wybiera CPU.
    """
    device_arg = device_arg.lower()

    if device_arg == "auto":
        # Wymuszenie CPU dla Maców (szybsze ewaluacje dla małych SNN)
        if sys.platform == "darwin":
            return "cpu"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    elif device_arg == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise RuntimeError(
                "[BŁĄD] Żądano urządzenia 'mps', ale PyTorch nie widzi środowiska "
                "Metal Performance Shaders (Apple Silicon)."
            )
        return "mps"

    elif device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("[BŁĄD] Żądano urządzenia 'cuda', ale sterowniki nie są dostępne.")
        return "cuda"

    elif device_arg == "cpu":
        return "cpu"

    else:
        raise ValueError(f"[BŁĄD] Nieznane urządzenie: {device_arg}. Wybierz: auto, cpu, cuda, mps.")


def _detect_worker_ceiling() -> int:
    """
    Liczba dostępnych rdzeni logicznych, z uwzględnieniem cgroup/taskset tam,
    gdzie to możliwe.

    os.sched_getaffinity nie istnieje na macOS (AttributeError) -- na M5 Max
    spadamy na os.cpu_count(), czyli hw.logicalcpu. Apple Silicon nie ma
    hyperthreadingu, więc logiczne == fizyczne; na dev-maszynie z HT liczba
    może być zawyżona względem realnej liczby rdzeni fizycznych -- to tylko
    ogranicznik sensownych wartości do testu, nie dowód optymalnego workera.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _default_worker_counts(ceiling: int | None = None) -> tuple[int, ...]:
    """1/2/4/8/12/16 ograniczone do faktycznie dostępnych rdzeni (wymóg M0)."""
    ceiling = ceiling if ceiling is not None else _detect_worker_ceiling()
    candidates = (1, 2, 4, 8, 12, 16)
    counts = tuple(w for w in candidates if w <= ceiling)
    return counts or (1,)


def _dummy_eval(worker_id: int, matrix_size: int = 800, iterations: int = 40) -> float:
    """
    Sztuczne zadanie obciążające procesor, wyłącznie do pomiaru narzutu puli
    procesów (nie do oceny jakości GA -- do tego służy realny benchmark na
    RealFitness, krok 3 M0).

    Operandy są stałe (a, b), a wynik każdej iteracji jest odrzucany po
    zredukowaniu do skalara: brak sprzężenia zwrotnego x = x @ x, które w
    oryginalnej wersji rozbiegało się do inf/nan już przy piątej iteracji na
    macierzy 800x800 (sprawdzone empirycznie) i zniekształcało pomiar czasu
    operacjami na wartościach niebędących liczbami skończonymi.
    """
    torch.set_num_threads(1)
    torch.manual_seed(worker_id)
    a = torch.randn(matrix_size, matrix_size)
    b = torch.randn(matrix_size, matrix_size)
    acc = 0.0
    for _ in range(iterations):
        c = a @ b
        acc += c.abs().mean().item()
    return acc


def _build_rf_kwargs_for_benchmark(config, project_root: str, device: str) -> dict:
    """
    Dokładnie te same argumenty co `run_ga_stage` w ga_runner.py -- benchmark
    ma mierzyć koszt tego samego RealFitness, którego użyje produkcyjny GA,
    a nie osobno wymyśloną konfigurację.

    fitness_seeds=3 jest tu przepisane na sztywno, bo tak samo jest dziś w
    run_ga_stage (nieparametryzowane przez config.train.fitness_seeds -- to
    osobny, znany błąd z M2, punkt 4). Naprawienie go tylko tutaj dałoby
    benchmark niezgodny z tym, co faktycznie uruchamia GA.
    """
    train_abs = os.path.join(project_root, config.data.train)
    val_abs = os.path.join(project_root, config.data.val)
    test_abs = os.path.join(project_root, config.data.test)
    arch_dir = os.path.dirname(os.path.dirname(train_abs))
    return dict(
        arch_dir=arch_dir,
        data=train_abs,
        val_data=val_abs,
        test_data=test_abs,
        limit=None,
        epochs=config.train.proxy_epochs,
        num_samples=config.train.num_samples,
        k=2,
        metric=config.ga.fitness_metric,
        fitness_seeds=3,
        pos_weight=1.0,
        feature_penalty=config.ga.feature_penalty,
        channels_head=None,
        stream_budget=6.0,
        stream_boot=0,
        verbose=False,
        seed=config.seed,
        device=device,
    )


def _run_real_ga_once(config, device: str, w: int, total_tasks: int) -> tuple[float, dict]:
    """
    Jedna generacja GA na `total_tasks` realnych genomach, przez RealFitness
    / ParallelFitness -- czyli dokładnie ten workload, o który prosi M0
    ("ten sam zestaw rzeczywistych genomów, epok, próbek i seedów"), zamiast
    syntetycznego mnożenia macierzy.

    generations=1 celowo: mierzymy koszt oceny jednej populacji, a nie
    zbieżność GA na wielu pokoleniach (to zmieniałoby total workload w sposób
    zależny od losowości selekcji, nie od liczby workerów).
    pop_size=total_tasks celowo zastępuje config.ga.pop_size: to jest dźwignia
    stałego workloadu z tej samej poprawki co w trybie lightweight -- liczba
    ocenionych genomów nie może zależeć od w.
    seed=config.seed jest identyczny dla każdego w, więc populacja startowa
    powinna być identyczna niezależnie od liczby workerów. Nie mam wglądu w
    kod run_ga/Genome, więc to jest założenie -- funkcja loguje best_fitness
    dla każdego w i ostrzega, jeśli się różnią, co byłoby sygnałem, że liczba
    workerów wpływa na to, co się liczy, a nie tylko na to, jak szybko.

    Zaimportowane leniwie (nie na górze pliku): main.py robi
    `from hardware import ...` przed `from ga_runner import ...`, a to
    ga_runner.py ustawia sys.path pod ga_neuron_search w swoim kodzie na
    poziomie modułu. Import na górze hardware.py wywaliłby się, zanim ten
    sys.path zdąży powstać.
    """
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for p in (project_root, os.path.join(project_root, "ga_neuron_search")):
            if p not in sys.path:
                sys.path.insert(0, p)
        from ga_neuron_search.fitness import RealFitness, ParallelFitness
        from ga_neuron_search.ga import GAConfig, run_ga
    except ImportError as exc:
        raise RuntimeError(
            "[BŁĄD] Benchmark w trybie 'real' wymaga ga_neuron_search "
            f"(RealFitness/ParallelFitness/GAConfig/run_ga), a import się nie udał: {exc}"
        ) from exc

    rf_kwargs = _build_rf_kwargs_for_benchmark(config, project_root, device)
    max_neurons = max(config.ga.neurons_range)

    # config.ga.elite jest dobrany do config.ga.pop_size (produkcyjne 30),
    # nie do total_tasks (u nas mniejsze na potrzeby benchmarku/smoke testu).
    # Bez tego ograniczenia elite > pop_size (np. elite=3 przy total_tasks=2)
    # prosi GA o wybranie większej liczby elit niż jest osobników w populacji.
    effective_elite = min(config.ga.elite, total_tasks)
    if effective_elite != config.ga.elite:
        print(
            f"  [UWAGA] w={w}: config.ga.elite={config.ga.elite} > total_tasks={total_tasks}; "
            f"przycięto do elite={effective_elite} dla tego benchmarku (nie dotyka config.json)."
        )

    ga_cfg = GAConfig(
        n_total=max_neurons,
        pop_size=total_tasks,
        generations=1,
        elite=effective_elite,
        max_hidden_layers=4,
        seed=config.seed,
    )

    fitness_evaluator = ParallelFitness(rf_kwargs, max_workers=w) if w > 1 else RealFitness(**rf_kwargs)
    start = time.perf_counter()
    try:
        res = run_ga(fitness_evaluator, ga_cfg, log=lambda *a, **k: None)
    finally:
        if hasattr(fitness_evaluator, "close"):
            fitness_evaluator.close()
    elapsed = time.perf_counter() - start

    # UWAGA: res.evaluated zwykle NIE równa się total_tasks (pop_size) dosłownie --
    # przy generations=1 run_ga ocenia populację startową (pop_size) i dokłada
    # potomków z jednej tury ewolucji (pop_size - elite), więc np. dla
    # pop_size=4, elite=3 wychodzi 4 + (4-3) = 5. To zaobserwowane empirycznie
    # (smoke test 24.09.2026), nie potwierdzone w źródle ga_neuron_search/ga.py.
    # Ważna jest tu spójność MIĘDZY konfiguracjami workerów (sprawdzana niżej w
    # benchmark_workers), nie zgodność z total_tasks wprost.

    return elapsed, {"best_fitness": res.best.fitness, "evaluated": res.evaluated}


def benchmark_workers(
    worker_counts: tuple[int, ...] | None = None,
    total_tasks: int = 64,
    repeats: int = 3,
    warmup: bool = True,
    config=None,
    device: str = "cpu",
) -> tuple[int, dict]:
    """
    Benchmark liczby workerów przy stałym, identycznym workloadzie w każdej
    konfiguracji (poprzednio: total_tasks = w * tasks_per_worker, więc więcej
    workerów dostawało więcej pracy i porównanie czasu nic nie mówiło o
    narzucie).

    Dwa tryby:
    - `config=None` (domyślny, wsteczna zgodność): syntetyczne zadanie
      (bounded matmul) mierzące wyłącznie narzut ProcessPoolExecutor. Dobre
      do szybkiego sanity-checku multiprocessingu bez datasetu, ale NIE
      spełnia wymogu M0 z rzeczywistymi genomami.
    - `config=<PipelineConfig>`: każda konfiguracja ocenia `total_tasks`
      rzeczywistych genomów przez RealFitness/ParallelFitness z tymi samymi
      epokami, próbkami i seedem co produkcyjny run_ga_stage. To jest tryb,
      którego wymaga M0.

    Kontekst multiprocessing (tryb lightweight) jest jawnie ustawiony na
    "spawn" -- domyślne dla macOS, ale nie dla Linuksa (tam domyślnie
    "fork"), więc bez tego pomiar na maszynie deweloperskiej mierzyłby inny
    mechanizm niż produkcyjny na Macu. W trybie real kontrolę nad pulą
    procesów ma ParallelFitness, więc tej funkcji nie zarządzamy tutaj
    osobnym ProcessPoolExecutorem.
    """
    if worker_counts is None:
        worker_counts = _default_worker_counts()

    mode = "real (RealFitness/GA, 1 generacja)" if config is not None else "lightweight (syntetyczny matmul)"
    print(
        f"[HARDWARE] Benchmark liczby workerów: {worker_counts}; tryb: {mode}; "
        f"stały workload = {total_tasks} {'genomów' if config is not None else 'zadań'} "
        f"na konfigurację, {repeats} powtórzeń" + (" + warmup" if warmup else "") + "."
    )
    if config is None:
        print(
            "[UWAGA] Bez `config` mierzony jest tylko narzut Pythona/"
            "ProcessPoolExecutor na syntetycznym zadaniu, nie realny koszt GA. "
            "To nie jest jeszcze odbiór M0 -- podaj `config`, żeby uruchomić tryb real."
        )

    results: dict[int, float] = {}
    diagnostics: dict[int, dict] = {}
    ctx = multiprocessing.get_context("spawn")

    for w in worker_counts:

        def _run_once() -> tuple[float, dict]:
            if config is not None:
                return _run_real_ga_once(config, device, w, total_tasks)
            start = time.perf_counter()
            with concurrent.futures.ProcessPoolExecutor(max_workers=w, mp_context=ctx) as executor:
                list(executor.map(_dummy_eval, range(total_tasks)))
            return time.perf_counter() - start, {}

        if warmup:
            _run_once()  # pomiar odrzucony: rozgrzewa pulę procesów, cache, load datasetu

        timed = [_run_once() for _ in range(repeats)]
        timings = [t for t, _ in timed]
        median_elapsed = statistics.median(timings)
        results[w] = round(median_elapsed, 3)
        if config is not None:
            diagnostics[w] = timed[-1][1]

        extra = ""
        if config is not None and diagnostics[w].get("best_fitness") is not None:
            extra = (
                f" | best_fitness={diagnostics[w]['best_fitness']:.4f}, "
                f"evaluated={diagnostics[w]['evaluated']}"
            )
        print(
            f"  -> {w} workerów: mediana {median_elapsed:.3f} s "
            f"(powtórzenia: {[round(t, 3) for t in timings]}){extra}"
        )

    if config is not None and diagnostics:
        fitness_values = {w: d.get("best_fitness") for w, d in diagnostics.items()}
        distinct_fitness = {round(v, 6) for v in fitness_values.values() if v is not None}
        if len(distinct_fitness) > 1:
            print(
                f"[UWAGA] best_fitness różni się między konfiguracjami workerów: "
                f"{fitness_values}. Przy identycznym seedzie i total_tasks oczekiwano "
                f"tego samego wyniku -- sprawdź, czy liczba workerów wpływa na "
                f"próbkowanie/kolejność wewnątrz run_ga/ParallelFitness."
            )

        evaluated_values = {w: d.get("evaluated") for w, d in diagnostics.items()}
        distinct_evaluated = {v for v in evaluated_values.values() if v is not None}
        if len(distinct_evaluated) > 1:
            print(
                f"[UWAGA] Liczba ocenionych genomów (evaluated) różni się między "
                f"konfiguracjami workerów: {evaluated_values}. To by znaczyło, że "
                f"workload jednak zależy od w -- sprawdź GAConfig/run_ga."
            )
        elif distinct_evaluated:
            (only_value,) = distinct_evaluated
            if only_value != total_tasks:
                print(
                    f"[INFO] evaluated={only_value} przy pop_size={total_tasks} -- "
                    f"identyczne dla każdego w (dobrze), ale różne od pop_size. "
                    f"Prawdopodobnie run_ga przy generations=1 dolicza potomków z "
                    f"jednej tury ewolucji (pop_size - elite); do potwierdzenia w "
                    f"źródle ga_neuron_search/ga.py, jeśli chcesz mieć pewność."
                )

    best_workers = min(results, key=results.get)
    print(
        f"[HARDWARE] Najszybsza konfiguracja: {best_workers} workerów "
        f"(mediana {results[best_workers]:.3f} s, identyczny workload {total_tasks} "
        f"{'genomów' if config is not None else 'zadań'} w każdej konfiguracji)."
    )
    if config is None:
        print(
            "[UWAGA] Wybór wyłącznie na podstawie narzutu procesów, bez pomiaru "
            "jakości i pamięci -- to nie jest jeszcze odbiór M0."
        )

    return best_workers, results


def resolve_workers(
    workers_arg,
    default_auto_options: tuple[int, ...] | None = None,
    config=None,
    device: str = "cpu",
) -> tuple[int, dict]:
    """
    Rozwiązuje parametr CLI określający liczbę workerów.
    Zwraca krotkę: (liczba_workerów, wyniki_benchmarku_jeśli_wykonano).

    `default_auto_options=None` pozwala benchmark_workers samemu wykryć
    dostępne rdzenie (_default_worker_counts) zamiast hardkodowanej listy.
    `config`/`device`, jeśli podane, włączają tryb real (RealFitness zamiast
    syntetycznego zadania) -- patrz benchmark_workers.
    """
    if str(workers_arg).lower() == "auto":
        return benchmark_workers(default_auto_options, config=config, device=device)

    try:
        w = int(workers_arg)
        if w <= 0:
            raise ValueError
        return w, {}
    except ValueError:
        raise ValueError(f"[BŁĄD] Flaga --workers musi być 'auto' lub >0, podano: {workers_arg}")
    