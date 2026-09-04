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


def _dummy_eval(worker_id: int) -> float:
    """
    Sztuczne zadanie obciążające procesor, symulujące proxy-trening w SNN.
    Służy wyłącznie do benchmarku overheadu multiprocessingowego.
    """
    # Zabezpieczenie rdzeni – zapobiega walce wątków PyTorch w tle
    torch.set_num_threads(1)
    
    # Symulacja krótkiego obciążenia tensorami
    x = torch.randn(800, 800)
    for _ in range(40):
        x = torch.matmul(x, x)
    return x.sum().item()


def benchmark_workers(worker_counts=(8, 10, 12, 14), tasks_per_worker=4) -> tuple[int, dict]:
    """
    Krótki benchmark badający narzut (overhead) w celu wyboru optymalnej
    liczby procesów równoległych dla konkretnej maszyny (np. M5 Pro).
    """
    print(f"[HARDWARE] Startuję benchmark liczby workerów dla: {worker_counts}...")
    results = {}

    for w in worker_counts:
        total_tasks = w * tasks_per_worker
        start_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(max_workers=w) as executor:
            # Mapujemy zadanie na pulę procesów; list() zmusza do wykonania
            list(executor.map(_dummy_eval, range(total_tasks)))

        elapsed = time.time() - start_time
        results[w] = round(elapsed, 3)
        print(f"  -> {w} workerów: {elapsed:.2f} s")

    best_workers = min(results, key=results.get)
    print(f"[HARDWARE] Zwycięzca: {best_workers} workerów (zoptymalizowany czas).\n")

    return best_workers, results


def resolve_workers(workers_arg, default_auto_options=(8, 10, 12, 14)) -> tuple[int, dict]:
    """
    Rozwiązuje parametr CLI określający liczbę workerów.
    Zwraca krotkę: (liczba_workerów, wyniki_benchmarku_jeśli_wykonano).
    """
    if str(workers_arg).lower() == "auto":
        return benchmark_workers(default_auto_options)
    
    try:
        w = int(workers_arg)
        if w <= 0:
            raise ValueError
        return w, {}
    except ValueError:
        raise ValueError(f"[BŁĄD] Flaga --workers musi być 'auto' lub >0, podano: {workers_arg}")
    